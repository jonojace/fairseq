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


def zeropad_to_len(t, targ_len, hid_dim):
    len_diff = targ_len - t.size(0)
    return torch.cat([t, t.new_zeros(len_diff, hid_dim)]), len_diff


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

    """

    def __init__(
            self,
            data_path,
            split,
            num_wordtypes=None,
            max_examples_per_wordtype=None,
    ):
        super().__init__()

        if max_examples_per_wordtype is not None:
            assert max_examples_per_wordtype >= 2, "2 examples needed to draw a positive example for a given anchor."

        ################################################################################################################
        # open main data folder and load word-aligned speech reps for all words in the vocab
        self.fpaths = []
        self.sizes = []
        all_indices = []

        # a mapping between a wordtype and a list of positive examples of that wordtype
        # this data structure is used to quickly find positive and negative examples for a particular word token
        self.wordtype2indices = defaultdict(set)

        # load all subfolders (which correspond to wordtypes)
        all_subfolders = sorted(os.listdir(data_path))
        # TODO add optional randomisation of this list of all wordtypes?
        # TODO so that if we choose a subset they are evenly distributed.

        # ignore some given wordtypes
        wordtypes_to_ignore = ['SIL', '<unk>']
        for w in wordtypes_to_ignore:
            if w in all_subfolders:
                all_subfolders.remove(w)

        # start adding each word token to the list of filenames, wordtype by wordtype
        idx = 0
        skipped_wordtypes = []
        logger.info(f"Creating dataset...")
        for wordtype in tqdm(all_subfolders[:num_wordtypes], unit='wordtype'):
            all_wordtoken_files = os.listdir(os.path.join(data_path, wordtype))
            if len(all_wordtoken_files) >= 2:
                for wordtoken_file in all_wordtoken_files[:max_examples_per_wordtype]:
                    filepath = os.path.join(data_path, wordtype, wordtoken_file)

                    # assign data associated with this word token / index
                    self.sizes.append(int(get_timesteps_from_filename(wordtoken_file)))
                    self.fpaths.append(filepath)
                    self.wordtype2indices[wordtype].add(idx)
                    all_indices.append(idx)

                    idx += 1
            else:
                skipped_wordtypes.append(wordtype)

        self.sizes = np.array(self.sizes, dtype=np.int64)
        # self.fpaths = pyarrow.array(self.fpaths)  # uncomment to increase performance using pyarrow

        assert all_indices == list(range(len(self.fpaths)))
        assert_msg = f"len(self.sizes)=={len(self.sizes)}, len(self.fnames)=={len(self.fpaths)}, sum(len(v) for v in " \
                     f"self.wordtype2indices.values())=={sum(len(v) for v in self.wordtype2indices.values())}, idx + 1=={idx} "
        assert len(self.sizes) == len(self.fpaths) == sum(
            len(v) for v in self.wordtype2indices.values()) == idx, assert_msg

        self.all_indices = set(all_indices)

        logger.info(f"Finished creating word-aligned speech representations dataset containing {len(self.fpaths)} "
                    "word tokens in total.")
        logger.info(f"Did not include {len(skipped_wordtypes)} wordtypes because they only have 1 example. At least 2 "
                    f"examples are required for triplet loss training.")

    def __getitem__(self, anchor_index):
        positive_index = list(self.get_positive_indices(anchor_index, num_examples=1))[0]
        negative_index = list(self.get_negative_indices(anchor_index, num_examples=1))[0]

        anchor_in = torch.load(self.fpaths[anchor_index])
        positive_in = torch.load(self.fpaths[positive_index])
        negative_in = torch.load(self.fpaths[negative_index])

        return {
            "anchor_index": anchor_index,
            "positive_index": positive_index,
            "negative_index": negative_index,
            "anchor_in": anchor_in,
            "positive_in": positive_in,
            "negative_in": negative_in,
        }

    def __len__(self):
        return len(self.fpaths)

    def index2wordtype(self, index):
        filepath = self.fpaths[index]
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

        # get indices
        anchor_indices = torch.tensor([s["anchor_index"] for s in samples], dtype=torch.long)
        positive_indices = torch.tensor([s["positive_index"] for s in samples], dtype=torch.long)
        negative_indices = torch.tensor([s["negative_index"] for s in samples], dtype=torch.long)

        # get speech representation inputs
        anchor_ins = [s["anchor_in"] for s in samples]
        positive_ins = [s["positive_in"] for s in samples]
        negative_ins = [s["negative_in"] for s in samples]

        # get timesteps of each input before performing zero padding for batching
        anchor_szs = torch.tensor([s["anchor_in"].size(0) for s in samples], dtype=torch.long)
        positive_szs = torch.tensor([s["positive_in"].size(0) for s in samples], dtype=torch.long)
        negative_szs = torch.tensor([s["negative_in"].size(0) for s in samples], dtype=torch.long)

        # TODO perform sorting according to length like in speech_to_text_dataset.py? how does this increase perf?

        ################################################################################################################
        # zero pad according to the longest sequence among anchor, positive, or negative inputs.

        # create 0s tensor
        b_sz = 3 * len(samples)
        max_len = torch.max(torch.cat([anchor_szs, positive_szs, negative_szs]))
        hid_dim = samples[0]["anchor_in"].size(1)
        collated_inputs = torch.zeros(b_sz, max_len, hid_dim)
        padding_mask = torch.BoolTensor(b_sz, max_len).fill_(True)
        # populate with data
        for i, (anchor_in, positive_in, negative_in) in enumerate(zip(anchor_ins, positive_ins, negative_ins)):
            collated_inputs[3 * i], anchor_len_diff = zeropad_to_len(anchor_in, max_len, hid_dim)
            collated_inputs[3 * i + 1], positive_len_diff = zeropad_to_len(positive_in, max_len, hid_dim)
            collated_inputs[3 * i + 2], negative_len_diff = zeropad_to_len(negative_in, max_len, hid_dim)
            padding_mask[3 * i, -anchor_len_diff:] = False
            padding_mask[3 * i + 1, -positive_len_diff:] = False
            padding_mask[3 * i + 2, -negative_len_diff:] = False

        print("end of loop")

        return {
            "collated_inputs": collated_inputs,
            "padding_mask": padding_mask,
            # "anchor_indices": anchor_indices,
            # "positive_indices": positive_indices,
            # "negative_indices": negative_indices,
            # "anchor_ins": anchor_ins,
            # "positive_ins": positive_ins,
            # "negative_ins": negative_ins,
            # "anchor_szs": anchor_szs,
            # "positive_szs": positive_szs,
            # "negative_szs": negative_szs,
        }

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


def test():
    logging.basicConfig(format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
    logger.setLevel(logging.INFO)

    # test dataset creation
    data_path = '/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/'
    dataset = WordAlignedAudioDataset(
        data_path,
        num_wordtypes=100,
        max_examples_per_wordtype=5,
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
    """python /home/s1785140/fairseq/fairseq/data/audio/word_aligned_audio_dataset.py"""
    test()
