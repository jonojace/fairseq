from fairseq.data import FairseqDataset
# from .. import FairseqDataset
import logging
import numpy as np
import os
import pyarrow
import re
import torch
import random
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger(__name__)


def verify_size(reps):
    # check dimensions and change them if necessary
    if reps.dim() == 3:
        # we assume (batch, timesteps, hidden_size)
        reps = reps.squeeze(0)  # squeeze away batch dimension
        assert reps.dim() == 2
    elif reps.dim() == 2:
        pass  # all is good!
    else:
        raise ValueError("speech representations have an incorrect number of dimensions")
    return reps


def get_timesteps_from_filename(filename):
    # grep out length
    matches = re.findall(r'len(\d+)', filename)
    assert len(matches) == 1
    return matches[0]


def random_sampling(a_list, num_samples):
    if num_samples >= len(a_list):
        return a_list
    else:
        return random.sample(a_list, num_samples)


def zeropad_to_len(t, targ_len):
    len_diff = targ_len - t.size(0)
    if t.dim() == 1:
        zero_padded_t = torch.cat([t, t.new_zeros(len_diff)])
    elif t.dim() == 2:
        zero_padded_t = torch.cat([t, t.new_zeros(len_diff, t.size(1))])
    else:
        raise ValueError
    return zero_padded_t, len_diff


class WordAlignedAudioDataset(FairseqDataset):
    """
    A dataset that maps between word-tokens in a corpus to their speech representations.

    Speech aligned at the word-level can be represented as any two dimensional matrix of shape (timesteps, dimensions).
    E.g.:
        - mel-spectrograms (timesteps, number of mel bins)
        - wav2vec2.0 representations (timesteps, hidden size of wav2vec2.0 representations)

    The dataset is structured as a key-value mapping:
        - key: An index from 0 to the total number of tokens in a corpus.
               Subsequently the index uniquely identifies any token in the corpus
        - value: A dictionary associated with that token. That contains:
            - path to a token's speech representations
            - the token's graphemes

    This dataset also contains a mapping between a word in its graphemic form and its examples in the corpus.
    This is used to speed up the retrieval of positive and negative examples for triplet loss/contrastive training.

    The data_path is the path to the speech corpus rearranged and cut at the word-level,
    it should have the following structure (please refer to fairseq/examples/lexicon_learner/wordalign_speechreps.py):
    - data_path
        - word1
            - word1_LJ010-0292_001.pt
            - word1_LJ010-0292_002.pt
            - ...
        - word2
            - word2_LJ001-0012_001.pt
            - word2_LJ002-0024_001.pt
            - ...
        - ...

    - word1, word2, ... subfolders refer to a particular wordtype in the corpus.
    - .pt files contain speech representations that map to a particular example of a wordtype.
      It is named as:
        <wordtype>_<utt id>_occ<numbered occurrence in the utterance>_len<num of timesteps in sequence>.pt

    Training set:

    Validation set:
        Seen:
        Unseen:

    """

    def __init__(
            self,
            data_path,
            split,
            save_dir,
            cache_all_data=True, # warning doing so with large datasets could lead to OOM!
            max_train_wordtypes=None, # leave as None to use as many wordtypes as possible for training
            max_train_examples_per_wordtype=None, # leave as None to use all examples for each wordtype
            min_train_examples_per_wordtype=2,
            valid_seen_wordtypes=100,  # how many wordtypes seen during training to include in validation
            valid_unseen_wordtypes=100,  # how many wordtypes to leave out of training and include in validation
            valid_examples_per_wordtype=25, # for valid-seen and valid-unseen
            randomise_wordtypes=True,
            random_seed=1337,
            wordtypes_to_ignore=('SIL', '<unk>'),
            debug_only_include_words_beginning_with=None,
            padding_index_offset=1,
    ):
        super().__init__()

        logger.info(f"Creating dataset...")

        # valid-seen is by definition a subset of the training dataset
        if max_train_wordtypes is not None and max_train_wordtypes < valid_seen_wordtypes:
            raise ValueError(f"max_train_wordtypes ({max_train_wordtypes}) < valid_seen_wordtypes ({valid_seen_wordtypes})")

        # need at least 2 examples for training and 2 for validation (2+2=4)
        # so that we can pull at least 1 positive example during training and validation for a wordtype
        assert min_train_examples_per_wordtype >= 2
        assert valid_examples_per_wordtype >= 2

        min_examples_per_wordtype = min_train_examples_per_wordtype + valid_examples_per_wordtype

        # if max_train_examples_per_wordtype is not None:
        #     assert max_train_examples_per_wordtype >= min_examples_per_wordtype, f"At least {min_examples_per_wordtype} examples needed to draw a positive example for a given anchor during either training or validation. (max_train_examples_per_wordtype={max_train_examples_per_wordtype})"

        # check data split
        if split == "test":
            raise NotImplementedError
        if split not in ["train", "valid-seen", "valid-unseen"]:
            raise ValueError(f"'{split}' not a correct dataset split.")

        ################################################################################################################
        ### Open main data folder and load word-aligned speech reps for all words in the vocab
        self.examples = [] # each example is a filepath to some reps, or the representations themselves
        self.sizes = []
        all_indices = []
        self.cache_all_data = cache_all_data
        self.padding_index_offset = padding_index_offset

        # create a mapping between a wordtype and a list of positive examples of that wordtype
        # this data structure is used to quickly find positive and negative examples for a particular word token
        self.wordtype2indices = defaultdict(set)

        # load all subfolders (each of which correspond to a unique wordtype)
        all_subfolders = sorted(os.listdir(data_path))

        # (for debugging) optionally only include wordtypes that start with a certain letter to speed up debugging
        if debug_only_include_words_beginning_with is not None:
            all_subfolders = [w for w in all_subfolders if w.startswith(debug_only_include_words_beginning_with)]

        # optionally randomise the order so its not alphabetical
        if randomise_wordtypes:
            random.seed(random_seed)
            random.shuffle(all_subfolders)

        # skip wordtypes we wish to ignore
        for w in wordtypes_to_ignore:
            if w in all_subfolders:
                all_subfolders.remove(w)

        # skip any wordtypes from consideration if they do not have enough examples
        skipped_wordtypes = []
        logger.info(f"Skipping wordtypes that do not have enough examples...")
        for wordtype in tqdm(all_subfolders, unit='wordtype'):
            all_wordtoken_files = os.listdir(os.path.join(data_path, wordtype))
            if len(all_wordtoken_files) < min_examples_per_wordtype:
                skipped_wordtypes.append(wordtype)
        for w in skipped_wordtypes:
            all_subfolders.remove(w)
        logger.info(f"Did not include {len(skipped_wordtypes)} wordtypes because they have fewer than {min_examples_per_wordtype} examples.")

        # calculate start and end wordtype indices depending on the dataset split/split subset
        if split == "train":
            start_wordtype_idx = 0
            if max_train_wordtypes is None:
                end_wordtype_idx = len(all_subfolders) - valid_unseen_wordtypes
            else:
                if len(all_subfolders) >= max_train_wordtypes + valid_unseen_wordtypes:
                    end_wordtype_idx = max_train_wordtypes
                else:
                    end_wordtype_idx = len(all_subfolders) - valid_unseen_wordtypes
        elif split == "valid-seen":
            start_wordtype_idx = 0
            end_wordtype_idx = valid_seen_wordtypes
        elif split == "valid-unseen":
            start_wordtype_idx = -valid_unseen_wordtypes
            end_wordtype_idx = None
        else:
            raise ValueError(f"'{split}' not a correct dataset split or dataset split subset.")

        wordtype_to_incl_idx = 0

        for wordtype in tqdm(all_subfolders[start_wordtype_idx:end_wordtype_idx], unit='wordtype'):
            all_wordtoken_files = os.listdir(os.path.join(data_path, wordtype))
            all_wordtoken_files = sorted(all_wordtoken_files) # ensure consistent ordering

            # calculate start and end wordtoken indices depending on the dataset split/split subset
            if split in ["train"]:
                start_wordtoken_idx = 0
                if max_train_examples_per_wordtype is None:
                    end_wordtoken_idx = len(all_wordtoken_files) - valid_examples_per_wordtype
                else:
                    if len(all_wordtoken_files) >= max_train_examples_per_wordtype + valid_examples_per_wordtype:
                        end_wordtoken_idx = max_train_examples_per_wordtype
                    else:
                        end_wordtoken_idx = len(all_wordtoken_files) - valid_examples_per_wordtype
            elif split in ["valid-seen", "valid-unseen"]:
                start_wordtoken_idx = -valid_examples_per_wordtype
                end_wordtoken_idx = None
            else:
                raise ValueError(f"'{split}' not a correct dataset split or dataset split subset.")

            for wordtoken_file in all_wordtoken_files[start_wordtoken_idx:end_wordtoken_idx]:
                filepath = os.path.join(data_path, wordtype, wordtoken_file)

                # assign data associated with this word token / index
                self.sizes.append(int(get_timesteps_from_filename(wordtoken_file)))
                self.examples.append(filepath)
                self.wordtype2indices[wordtype].add(wordtype_to_incl_idx)
                all_indices.append(wordtype_to_incl_idx)

                wordtype_to_incl_idx += 1

        # cache all data in order to avoid accessing disk (locally or network) during training
        # (more feasible when loading smaller data formats such as hubert codes vs full wav2vec2.0 vectors)
        if self.cache_all_data:
            self.cached_data = []
            logger.info(f"Caching {len(self.examples)} examples.")
            for fp in tqdm(self.examples):
                self.cached_data.append(torch.load(fp).int() + self.padding_index_offset)

        # Sanity checks
        assert all_indices == list(range(len(self.examples)))
        assert_msg = f"len(self.sizes)=={len(self.sizes)}, len(self.fnames)=={len(self.examples)}, sum(len(v) for v in " \
                     f"self.wordtype2indices.values())=={sum(len(v) for v in self.wordtype2indices.values())}, idx + 1=={wordtype_to_incl_idx} "
        assert len(self.sizes) == len(self.examples) == sum(
            len(v) for v in self.wordtype2indices.values()) == wordtype_to_incl_idx, assert_msg

        # Assign object params
        self.sizes = np.array(self.sizes, dtype=np.int64)
        self.all_indices = set(all_indices)
        # self.examples = pyarrow.array(self.examples)  # uncomment to increase performance using pyarrow

        # Print/save important information and stats about this dataset
        logger.info(f"Finished creating word-aligned speech representations {split} dataset containing {len(self.wordtype2indices)} wordtypes "
                    f"and {len(self.examples)} word tokens in total.")
        if split in ["valid-seen", "valid-unseen"]:
            logger.info(f"{split} wordtypes are: {' '.join(self.wordtype2indices.keys())}")
        self.save_wordtypes_to_disk(os.path.join(save_dir, f'{split}_{len(self.wordtype2indices.keys())}_wordtypes.csv'))



    def __getitem__(self, anchor_index):
        positive_index = list(self.get_positive_indices(anchor_index, num_examples=1))[0]
        negative_index = list(self.get_negative_indices(anchor_index, num_examples=1))[0]

        # load inputs
        if self.cache_all_data:
            anchor_in = self.cached_data[anchor_index]
            positive_in = self.cached_data[positive_index]
            negative_in = self.cached_data[negative_index]
        else:
            anchor_in = torch.load(self.examples[anchor_index]).int() + self.padding_index_offset # TODO warning this will not work with continuous reps!
            positive_in = torch.load(self.examples[positive_index]).int() + self.padding_index_offset
            negative_in = torch.load(self.examples[negative_index]).int() + self.padding_index_offset

        # create tensors for indicating where we want to output targets
        # e.g. 1 in timesteps where there is a grapheme and 0 where we do not have a grapheme
        #      (padding will be performed later by collater)
        anchor_tgt = torch.ones(self.get_tgt_len(anchor_index, units="graphemes"), dtype=torch.int)
        positive_tgt = torch.ones(self.get_tgt_len(positive_index, units="graphemes"), dtype=torch.int)
        negative_tgt = torch.ones(self.get_tgt_len(negative_index, units="graphemes"), dtype=torch.int)

        # print("debug", self.index2wordtype(anchor_index), anchor_tgt)

        return {
            "anchor_index": anchor_index,
            "positive_index": positive_index,
            "negative_index": negative_index,
            "anchor_in": anchor_in, # e.g. speech reps of the anchor word
            "positive_in": positive_in,
            "negative_in": negative_in,
            "anchor_tgt": anchor_tgt,
            "positive_tgt": positive_tgt,
            "negative_tgt": negative_tgt,
        }

    def get_tgt_len(self, wordtoken_index, units="graphemes"):
        """
        return the length of some metadata related to the wordtype associated with a wordtoken
        e.g. number of graphemes or phonemes of that wordtype
        """
        if units == "graphemes":
            tgt_len = len(self.index2wordtype(wordtoken_index))
        elif units == "phones":
            raise NotImplementedError
        else:
            raise ValueError

        return tgt_len

    def __len__(self):
        return len(self.examples)

    def index2wordtype(self, index):
        filepath = self.examples[index]
        wordtype = filepath.split('/')[-2]
        return wordtype

    def get_all_indices_for_wordtype(self, index):
        wordtype = self.index2wordtype(index)
        all_indices_for_wordtype = self.wordtype2indices[wordtype]
        return all_indices_for_wordtype

    def get_positive_indices(self, anchor_index, num_examples=1):
        """
        For a given word token indicated by its index
        return a set of the indices of positive examples (word tokens of the SAME wordtype but not the same token!)
        """
        all_indices_for_wordtype = self.get_all_indices_for_wordtype(anchor_index)
        positive_indices = all_indices_for_wordtype - {anchor_index}
        return random_sampling(positive_indices, num_examples)

    def get_negative_indices(self, anchor_index, num_examples=1):
        """
        For a given word token indicated by its index
        return a set of the indices of negative examples (word tokens of a DIFFERENT wordtype)
        """
        all_indices_for_anchor = set(self.get_all_indices_for_wordtype(anchor_index))
        negative_indices = self.all_indices - all_indices_for_anchor
        return random_sampling(negative_indices, num_examples)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model

        Acoustic lexicon learner specific:
            - Zero pads anchor, positive, and negative inputs so that all items in the batch are the same size
        """
        if len(samples) == 0:
            return {}

        # # get indices
        # anchor_indices = torch.tensor([s["anchor_index"] for s in samples], dtype=torch.long)
        # positive_indices = torch.tensor([s["positive_index"] for s in samples], dtype=torch.long)
        # negative_indices = torch.tensor([s["negative_index"] for s in samples], dtype=torch.long)

        # # get wordtypes just for anchor words
        # anchor_wordtypes = [self.index2wordtype(idx) for idx in anchor_indices]

        collated_inputs, in_lengths = self.collate_anchor_positive_negative_tensors(samples, feat_type='in')
        collated_tgts, tgt_lengths = self.collate_anchor_positive_negative_tensors(samples, feat_type='tgt')

        return {
            "net_input": {
                "src_tokens": collated_inputs,
                "tgt_tokens": collated_tgts,
                # "src_lengths": in_lengths,
                "tgt_lengths": tgt_lengths,
            },
            "sample_size": in_lengths.sum().item(),
            # "anchor_indices": anchor_indices,
            # "positive_indices": positive_indices,
            # "negative_indices": negative_indices,
            # "anchor_wordtypes": anchor_wordtypes,
            # "nsentences": len(samples),
            # "ntokens": sum(len(s["source"]) for s in samples),
            # "target": target,
        }

    def collate_anchor_positive_negative_tensors(self, samples, feat_type):
        """
        collate anchor positive and negative for a given feat in samples

        feat_type could be:
            "in" for inputs
            "tgt" for targets
        for example
        """
        # get speech representation inputs, and place in lists
        anchor_tensors = [s[f"anchor_{feat_type}"] for s in samples]
        positive_tensors = [s[f"positive_{feat_type}"] for s in samples]
        negative_tensors = [s[f"negative_{feat_type}"] for s in samples]

        # zero pad according to the longest sequence among anchor, positive, or negative inputs.
        # get timesteps of each input before performing zero padding for batching
        # so that we know what the maximum length of the tensor should be
        anchor_szs = torch.tensor([s[f"anchor_{feat_type}"].size(0) for s in samples], dtype=torch.long)
        positive_szs = torch.tensor([s[f"positive_{feat_type}"].size(0) for s in samples], dtype=torch.long)
        negative_szs = torch.tensor([s[f"negative_{feat_type}"].size(0) for s in samples], dtype=torch.long)
        max_len = torch.max(torch.cat([anchor_szs, positive_szs, negative_szs])).item()
        # create 0s tensor
        b_sz = 3 * len(samples)
        if samples[0][f"anchor_{feat_type}"].dim() == 1:  # discrete
            collated_tensor = torch.zeros(b_sz, max_len, dtype=torch.int)
        elif samples[0][f"anchor_{feat_type}"].dim() == 2:  # continuous
            hid_dim = samples[0][f"anchor_{feat_type}"].size(1)
            collated_tensor = torch.zeros(b_sz, max_len, hid_dim, dtype=torch.float)
        else:
            raise ValueError
        # print("collated_tensor.size()", collated_tensor.size())
        lengths = torch.zeros(b_sz, dtype=torch.int64)

        # populate with data, group by anchors, positives, negatives
        for i, anchor_t in enumerate(anchor_tensors):
            # print("anchor", i)
            collated_tensor[i], _ = zeropad_to_len(anchor_t, max_len)
            lengths[i] = anchor_t.size(0)
        for i, positive_t in enumerate(positive_tensors, start=len(samples)):
            # print("positive", i)
            collated_tensor[i], _ = zeropad_to_len(positive_t, max_len)
            lengths[i] = positive_t.size(0)
        for i, negative_t in enumerate(negative_tensors, start=2*len(samples)):
            # print("negative", i)
            collated_tensor[i], _ = zeropad_to_len(negative_t, max_len)
            lengths[i] = negative_t.size(0)

        return collated_tensor, lengths

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self), dtype=np.int64)
    
    def save_wordtypes_to_disk(self, save_path):
        """Save wordtypes in this datasplit to disk"""
        with open(save_path, 'w') as f:
            f.write(','.join(self.wordtype2indices.keys()))
        logger.info(f"Successfully saved wordtypes to '{save_path}'")

def test():
    logging.basicConfig(format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
    logger.setLevel(logging.INFO)

    # test dataset creation
    data_path = '/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/'
    dataset = WordAlignedAudioDataset(
        data_path,
        max_train_wordtypes=100,
        max_train_examples_per_wordtype=5,
    )

    # for i in range(len(dataset)):
    #     print(dataset.index2wordtype(i))

    def print_words_for_indices(indices):
        print(", ".join(f"{j}:{dataset.index2wordtype(j)}" for j in indices))

    # test retrieval of +ve and -ve indices
    print_words_for_indices(dataset.get_positive_indices(anchor_index=0, num_examples=10))
    print_words_for_indices(dataset.get_negative_indices(anchor_index=0, num_examples=10))

    list_of_samples = []

    # test __getitem__()
    for i, sample in enumerate(dataset):
        print(sample)
        list_of_samples.append(sample)
        if i > 5:
            break

    # test collater
    collate_rv = dataset.collater(list_of_samples)

    print(collate_rv['anchor_indices'])


if __name__ == '__main__':
    """
    to test, run:
        python /home/s1785140/fairseq/fairseq/data/audio/word_aligned_audio_dataset.py
    """
    test()
