from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
import torch

from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset, SpeechToTextDatasetCreator, S2TDataConfig,
    _collate_frames, get_features_or_waveform
)

from fairseq.data.audio.text_to_speech_dataset import (
    TextToSpeechDatasetItem,
    TextToSpeechDataset,
    TextToSpeechDatasetCreator,
)

from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)

from fairseq.data.audio.speech_audio_corrector_utils import (
    get_mfa_text,
    get_word_pos,
    get_word2speechreps,
    get_speechreps_for_utt,
    prepend_speechreps_for_dict_encoding,
    two_random_partitions,
    mask_according_to_word_pos,
    get_text_inputs,
    get_speechreps_inputs,
    get_tokens,
    randomly_mask_words,
)

import os
import pickle

import textgrid
from collections import Counter
from fairseq.tokenizer import tokenize_line

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

@dataclass
class SpeechAudioCorrectorDatasetItem(TextToSpeechDatasetItem):
    word_pos: Optional[List[int]] = None
    text_len: int = -1
    speechreps_len: int = -1
    segment: torch.Tensor = None
    words_and_speechreps: List[Tuple] = None
    raw_text: str = None

class SpeechAudioCorrectorDataset(TextToSpeechDataset):
    def __init__(
            self,
            args,
            split: str,
            is_train_split: bool,
            cfg: S2TDataConfig,
            audio_paths: List[str],
            n_frames: List[int],
            src_texts: Optional[List[str]] = None,
            tgt_texts: Optional[List[str]] = None,
            speakers: Optional[List[str]] = None,
            src_langs: Optional[List[str]] = None,
            tgt_langs: Optional[List[str]] = None,
            ids: Optional[List[str]] = None,
            tgt_dict: Optional[Dictionary] = None,
            pre_tokenizer=None,
            bpe_tokenizer=None,
            n_frames_per_step=1,
            speaker_to_id=None,
            durations: Optional[List[List[int]]] = None,
            pitches: Optional[List[str]] = None,
            energies: Optional[List[str]] = None,
            word2speechreps: Optional[Dictionary] = None,
            ext_word2speechreps: Optional[Dictionary] = None,
            ids2word_alignments: Optional[Dictionary] = None,
            use_ext_word2speechreps_p: Optional[float] = 0.0,
    ):
        """

        Speech Audio Corrector (SAC) dataset

        Provides TTS inputs (text) and targets (speech)
        Also provides additional inputs for Speech Audio Corrector training. I.e. Speech representations corresponding to
        the utterance that are concatenated to the end of the text inputs during training to provide audio info for
        pronunciation corrections:
            - In the simplest case is simply identical to the TTS targets (e.g. mel-spectrogram)
            - Can be self-supervised speech representations for the utterance
            - Can be self-supervised speech representations pulled randomly from different utterances

        Effectively the same as a standard TTS dataset but with additional data for each utterance:
        - mapping from wordtype to speechrep codes for all examples of that word type (all examples because we want to
          optionally shuffle word examples during training to make SAC more robust

        Options:
        - remove duplicate codes
        - provide run length after removing duplicate codes (for run length encoding)
        """
        super(SpeechAudioCorrectorDataset, self).__init__(
            split, is_train_split, cfg, audio_paths, n_frames,
            src_texts=src_texts, tgt_texts=tgt_texts, speakers=speakers,
            src_langs=src_langs, tgt_langs=tgt_langs, ids=ids,
            tgt_dict=tgt_dict, pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer, n_frames_per_step=n_frames_per_step,
            speaker_to_id=speaker_to_id,
            durations=durations, pitches=pitches, energies=energies,
        )

        ################################################################################################################
        # add SAC specific CLAs from arg parser to cfg
        self.randomise_examples = args.randomise_examples
        self.use_ext_word2speechreps_p = args.use_ext_word2speechreps_p

        ################################################################################################################
        # add SAC specific data structure to this dataset object
        self.word_pos_pad_idx = 0
        self.segment_pad_idx = 0
        self.ids2word_alignments = ids2word_alignments
        self.word2speechreps = word2speechreps
        self.ext_word2speechreps = ext_word2speechreps

    def __getitem__(self, index: int) -> SpeechAudioCorrectorDatasetItem:
        ################################################################################################################
        # Grab standard text to speech dataset item
        t2s = super().__getitem__(index)
        utt_id = self.ids[index]

        ################################################################################################################
        # Text
        # use mfa words as input text rather than tgt_text as it perfectly matches our alignments and therefore
        # are easier to align to our speech reps. (tgt_text contains symbols and capitalisation that is not
        # taken into account by mfa
        mfa_text = get_mfa_text(self.ids2word_alignments[utt_id])
        masked_text = randomly_mask_words(mfa_text)
        sac_dataset_item = self.get_dataset_item_from_utt(
            masked_text,
            utt_id=utt_id,
            randomise=self.randomise_examples,
            index=index,
            source=t2s.source,
            speaker_id=t2s.speaker_id,
            bpe_token_delimiter=" ",
            use_external_speechreps=False,
        )

        return sac_dataset_item

    def get_dataset_item_from_utt(self,
                                  utt: str,
                                  utt_id=None,
                                  randomise=False,
                                  single_mask_tok_per_masked_word=True,
                                  index=None,
                                  source=None,
                                  speaker_id=None,
                                  bpe_token_delimiter=" ",
                                  use_external_speechreps=False,
                                  ) -> SpeechAudioCorrectorDatasetItem:
        ################################################################################################################
        # Process text into tokens
        # each token is a sequence of graphemes and has 1) word position 2) info about whether it is masked or not
        tokens = get_tokens(utt, padding_word_pos=self.word_pos_pad_idx)

        ################################################################################################################
        # Process tokens into graphemes and word positions for each grapheme
        graphemes, word_pos_of_graphemes = get_text_inputs(tokens, mask_token="<mask>",
                                                           single_mask_tok_per_masked_word=single_mask_tok_per_masked_word)

        ################################################################################################################
        # Process tokens into speech codes and word positions for each speech code

        # if use_external_speechreps:
        #     print("!!!DEBUGGG using self.ext_word2speechreps!!!")
        #     w2sr = self.ext_word2speechreps
        #     # TODO add option to combine the two dictionaries together???
        #     # do this in dataset creator when the two dictionaries are created
        # else:
        #     w2sr = self.word2speechreps

        speechreps, word_pos_of_speechreps = get_speechreps_inputs(
            tokens, self.word2speechreps,
            ext_word2speechreps=self.ext_word2speechreps,
            use_ext_word2speechreps_p=self.use_ext_word2speechreps_p,
            utt_id=utt_id, randomise=randomise
        )
        speechreps = prepend_speechreps_for_dict_encoding(speechreps)

        ################################################################################################################
        # Encode graphemes and speech reps
        # encode text into int indices, one for each grapheme (incl. whitespace)
        encoded_text = self.tgt_dict.encode_line(
            bpe_token_delimiter.join(graphemes), add_if_not_exist=False, append_eos=False
        ).long()
        assert len(word_pos_of_graphemes) == len(graphemes), f"{len(word_pos_of_graphemes)} != {len(graphemes)}"
        assert len(word_pos_of_graphemes) == len(encoded_text), f"{len(word_pos_of_graphemes)} != {len(encoded_text)}"

        # encode hubert codes into int indices, one for each hubert code
        encoded_speechreps = self.tgt_dict.encode_line(
            bpe_token_delimiter.join(speechreps), add_if_not_exist=False, append_eos=False
        ).long()
        assert len(word_pos_of_speechreps) == len(speechreps), f"{len(word_pos_of_speechreps)} != {len(speechreps)}"
        assert len(word_pos_of_speechreps) == len(
            encoded_speechreps), f"{len(word_pos_of_speechreps)} != {len(encoded_speechreps)}"

        # print(f"\n=== raw text :", utt)
        # print("***tokens")
        # print("tokens:", tokens)
        # print("***graphemes")
        # print("graphemes:", graphemes)
        # print("encoded_text:", encoded_text)
        # print("word_pos_of_graphemes:", word_pos_of_graphemes)
        # print("***speechreps")
        # print("speechreps:", speechreps)
        # print("encoded_speechreps:", encoded_speechreps)
        # print("word_pos_of_speechreps:", word_pos_of_speechreps)
        ################################################################################################################
        # Get length information for text and speechreps
        text_len = len(encoded_text)
        speechreps_len = len(encoded_speechreps)
        ################################################################################################################
        # Concat encoded text to encoded hubert codes for Speech Audio Corrector transformer inputs
        target = torch.cat([encoded_text, encoded_speechreps])
        word_pos_all_timesteps = torch.LongTensor(word_pos_of_graphemes + word_pos_of_speechreps)
        ################################################################################################################
        # Create segment information ids for looking up embeddings (1 is text, 2 is speechrep, 0 is padding)
        segment = torch.zeros(target.size()).long()
        segment[:text_len] = self.segment_pad_idx + 1  # if pad_idx == 0, then text gets idx 1
        segment[text_len:] = self.segment_pad_idx + 2  # if pad_idx == 0, then speechreps gets idx 2

        # TODO implement multi-speaker generation
        # if self.speaker_to_id is not None:
        #     speaker_id = self.speaker_to_id[self.speakers[index]]

        return SpeechAudioCorrectorDatasetItem(
            index=index,
            source=source, # mel spectrogram audio frames
            target=target, # graphemes concatenated with speech codes + token pos + word pos + segment info
            speaker_id=speaker_id,
            word_pos=word_pos_all_timesteps,
            text_len=text_len,
            speechreps_len=speechreps_len,
            segment=segment,
            raw_text=utt,
            # words_and_speechreps=words_and_speechreps,
        )

    def collater(self, samples: List[SpeechAudioCorrectorDatasetItem]) -> Dict[str, Any]:
        if len(samples) == 0:
            return {}

        ################################################################################################################
        # Text and speechreps inputs

        # sort items in batch by len of graphemes+speechreps
        src_lengths, order = torch.tensor(
            [s.target.shape[0] for s in samples], dtype=torch.long
        ).sort(descending=True)

        # inputs (graphemes concatenated with speechreps for speech audio corrector training)
        src_tokens = fairseq_data_utils.collate_tokens(
            [s.target for s in samples],
            self.tgt_dict.pad(),
        ).index_select(0, order)

        # word positions
        src_word_pos = fairseq_data_utils.collate_tokens(
            [s.word_pos for s in samples],
            self.word_pos_pad_idx,
        ).index_select(0, order)

        # segments
        src_segments = fairseq_data_utils.collate_tokens(
            [s.segment for s in samples],
            self.segment_pad_idx,
        ).index_select(0, order)

        ################################################################################################################
        # Misc.

        # speaker id
        speaker = None
        if self.speaker_to_id is not None:
            speaker = torch.tensor(
                [s.speaker_id for s in samples], dtype=torch.long
            ).index_select(0, order).view(-1, 1)

        # get human readable version of TTS inputs
        # examining this is a good way of double checking what the model sees as input,
        # and verify it doesn't cheat or get more information than it should
        # i.e. what words are graphemes, what words are hubert codes.
        src_texts = [self.tgt_dict.string(samples[i].target, include_eos=True) for i in order]
        raw_texts = [samples[i].raw_text for i in order]

        ##############################################
        # some features set to None as they are not
        # necessary for synthesis from text utts
        return_dict = {
            "id": None,
            "net_input": {
                "src_tokens": src_tokens,
                "src_word_pos": src_word_pos,
                "src_segments": src_segments,
                "src_lengths": src_lengths,
                "prev_output_tokens": None,
            },
            "speaker": speaker,
            "target": None,
            "target_lengths": None,
            "ntokens": None,
            "nsentences": len(samples),
            "raw_texts": raw_texts,
            "src_texts": src_texts,
            "words_and_speechreps": None,
        }

        ##############################################
        # add features needed for training
        if samples[0].index is not None:
            id_ = torch.tensor([s.index for s in samples],
                               dtype=torch.long).index_select(0, order)
            return_dict["id"] = id_

        if samples[0].words_and_speechreps is not None:
            words_and_speechreps = [samples[i].words_and_speechreps for i in order]
            return_dict["words_and_speechreps"] = words_and_speechreps

        if samples[0].source is not None:
            audio_feat = _collate_frames(
                [s.source for s in samples], self.cfg.use_audio_input
            ).index_select(0, order)

            # get audio original lens before zero padding
            target_lengths = torch.tensor(
                [s.source.shape[0] for s in samples], dtype=torch.long
            ).index_select(0, order)

            # get time-shifted audio for teacher forced training of TTS decoder
            bsz, _, d = audio_feat.size()
            prev_output_tokens = torch.cat(
                (audio_feat.new_zeros((bsz, 1, d)), audio_feat[:, :-1, :]), dim=1
            )

            return_dict["target"] = audio_feat
            return_dict["net_input"]["prev_output_tokens"] = prev_output_tokens
            return_dict["target_lengths"] = target_lengths
            return_dict["ntokens"] = sum(target_lengths).item()

        return return_dict

    def batch_from_utts(self,
                        all_utts,
                        dataset,
                        batch_size,
                        use_external_speechreps):
        """
        generate a batch of SAC model inputs from a list of plain text inputs

        Note: this function is used by inference code in order to easily get model inputs
        for test utts defined in arbitrary txt file

        all_utts: a list of text string utterances. words to be replaced by speech codes are in pointy brackets

        return:
            a batch of inputs ready for feeding into SAC
        """
        batches = [] # list of batches for input to SAC model
        utt_subsets = chunks(all_utts, batch_size)
        for utt_subset in utt_subsets:
            batch = []  # list of samples
            for utt in utt_subset:
                sample = self.get_dataset_item_from_utt(utt,
                                                        dataset,
                                                        use_external_speechreps=use_external_speechreps)
                batch.append(sample)
            batches.append(batch)
        collated_batches = [self.collater(batch) for batch in batches]
        return collated_batches

class SpeechAudioCorrectorDatasetCreator(TextToSpeechDatasetCreator):
    @classmethod
    def from_tsv(
            cls,
            args,
            root: str,
            cfg: S2TDataConfig,
            splits: str,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split: bool,
            epoch: int,
            seed: int,
            n_frames_per_step: int = 1,
            speaker_to_id=None
    ) -> SpeechToTextDataset:
        datasets = [
            cls._from_tsv(
                args, root, cfg, split, tgt_dict, is_train_split, pre_tokenizer,
                bpe_tokenizer, n_frames_per_step, speaker_to_id
            )
            for split in splits.split(",")
        ]

        if is_train_split and len(datasets) > 1 and cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls.get_size_ratios(datasets, alpha=cfg.sampling_alpha)
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for r, d in zip(size_ratios, datasets)
            ]

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    @classmethod
    def _from_tsv(
            cls,
            args,
            root: str,
            cfg: S2TDataConfig,
            split: str,
            tgt_dict,
            is_train_split: bool,
            pre_tokenizer,
            bpe_tokenizer,
            n_frames_per_step,
            speaker_to_id
    ) -> SpeechToTextDataset:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            args, split, is_train_split, samples, cfg, tgt_dict, pre_tokenizer,
            bpe_tokenizer, n_frames_per_step, speaker_to_id
        )

    @classmethod
    def _from_list(
        cls,
        args,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        n_frames_per_step,
        speaker_to_id,
    ) -> SpeechAudioCorrectorDataset:
        """
        Create a SAC dataset and return it
        """
        ################################################################################################################
        # get data needed to instantiate TTS dataset
        audio_paths, durations, energies, ids, n_frames, pitches, speakers, src_langs, src_texts, tgt_langs, tgt_texts = cls.get_tts_dataset_data(
            cfg, samples)

        ################################################################################################################
        # get SAC required data
        # dictionary mapping from a wordtype to speech codes for each word example in a corpus
        cls.word2speechreps_dir = "/home/s1785140/data/word2speechreps"

        # LJSpeech word2speechreps
        speechrep_file = "/home/s1785140/fairseq/examples/speech_audio_corrector/lj_speech_quantized.txt"
        alignments_dir = "/home/s1785140/data/ljspeech_MFA_alignments_from_fb"
        word2speechreps, ids2word_alignments = cls.get_word2speechreps(
            speechrep_file, alignments_dir, ids=ids, corpus="ljspeech", split=split_name, force_creation=False,
        )

        # VCTK word2speechreps
        # additional external source of word-aligned speech reps
        # (potentially for more robust multi-speaker corrections)
        ext_speechrep_file = "/home/s1785140/fairseq/examples/speech_audio_corrector/vctk_quantized.txt"
        ext_alignments_dir = "/home/s1785140/data/vctk_montreal_alignments_from_trimmed_wavs_no_nested_dirs"
        ext_word2speechreps, _ = cls.get_word2speechreps(
            ext_speechrep_file, ext_alignments_dir, ids=None, corpus="vctk", split=None, force_creation=False,
        )

        sac_dataset = SpeechAudioCorrectorDataset(
            args, split_name, is_train_split, cfg, audio_paths, n_frames,
            src_texts=src_texts, tgt_texts=tgt_texts, speakers=speakers,
            src_langs=src_langs, tgt_langs=tgt_langs, ids=ids,
            tgt_dict=tgt_dict, pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer, n_frames_per_step=n_frames_per_step,
            speaker_to_id=speaker_to_id,
            durations=durations, pitches=pitches, energies=energies,
            word2speechreps=word2speechreps, ext_word2speechreps=ext_word2speechreps,
            ids2word_alignments=ids2word_alignments,
        )

        return sac_dataset

    @classmethod
    def load_speechreps(cls, speechrep_file):
        with open(speechrep_file, 'r') as f:
            lines = f.readlines()
        utt_id2speechreps = {}
        for l in lines:
            utt_id, codes = l.split('|')
            codes = codes.rstrip()  # strip trailing newline char
            codes = [int(s) for s in codes.split(' ')]  # convert from str of ints to list of ints
            utt_id2speechreps[utt_id] = codes
        return utt_id2speechreps

    @classmethod
    def load_word_alignments(cls, ids, alignments_dir):
        utt_id2word_alignments = {}
        for utt_id in ids:
            words_align = cls.get_word_alignments(
                textgrid_path=f"{alignments_dir}/{utt_id}.TextGrid",
                utt_dur_from_last_word=False,
            )
            utt_id2word_alignments[utt_id] = words_align
        return utt_id2word_alignments

    @classmethod
    def get_word2speechreps(cls, speechrep_file, alignments_dir, ids, corpus, split, force_creation=False):
        """
        fast way to force recreation of data structures is by deleting them from disk @ cls.word2speechreps_dir
        """
        # see if existing word2speechreps exists for corpus
        prepend_str = corpus
        if split is not None:
            prepend_str = prepend_str + f"_{split}"
        filename = f"{prepend_str}_word2speechreps.pickle"
        filepath = os.path.join(cls.word2speechreps_dir, filename)

        if os.path.isfile(filepath) and not force_creation:
            # load dict
            with open(filepath, "rb") as f:
                word2speechreps, ids2word_alignments = pickle.load(f)
            print(f"Finished loading word2speechreps for {corpus} {split}. Loaded from {filepath}.")
        else:
            if force_creation:
                print(f"Forced recreation of word2speechreps for {corpus} {split}. Creating...")
            else:
                print(f"word2speechreps for {corpus} {split} not found. Creating...")
            # create dict
            ids2speechreps = cls.load_speechreps(speechrep_file)
            if ids is None:
                # if ids is None then create word2speech reps for the words in all the corpus
                # if ids is not None create it for only the words in the specified utterance ids
                ids = list(ids2speechreps.keys())
            ids2word_alignments = cls.load_word_alignments(ids, alignments_dir)
            word2speechreps = get_word2speechreps(ids, ids2speechreps, ids2word_alignments)
            # save dict
            with open(filepath, "wb") as f:
                to_dump = (word2speechreps, ids2word_alignments)
                pickle.dump(to_dump, f)
            print(f"Finished creating word2speechreps for {corpus} {split}. Saved to {filepath}.")
        return word2speechreps, ids2word_alignments

    @classmethod
    def get_word_alignments(
        cls,
        textgrid_path,
        utt_dur_from_last_word=False,
        ignore_list=('<unk>',),
    ):
        """
        extract word alignments from textgrid file corresponding to one utterance

        utt_dur_from_last_word: whether to set utt_dur to end timestamp of  last real wordtype, or from
        the very last alignment in the utterance (likely corresponding to silence)
        """
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        words_intervaltier, _phones_intervaltier = tg
        words = []
        counter = Counter()

        for word in words_intervaltier:
            if word.mark and word.mark not in ignore_list:  # if word.mark is False then it is SILENCE
                counter[word.mark] += 1
                words.append({
                    "wordtype": word.mark,
                    "utt_id": textgrid_path.split('/')[-1].split('.')[0],
                    "example_no": counter[word.mark],  # the number of times we have seen this word in this utterance
                    "start": word.minTime,
                    "end": word.maxTime,
                })

        if utt_dur_from_last_word:
            # use last real word end time as the utt_dur
            utt_dur = words[-1]['end']
        else:
            # at this point word is the last item in words_intervaltier (most likely sil / None)
            utt_dur = word.maxTime

        # add utt_dur info to all words
        for w in words:
            w["utt_dur"] = utt_dur

        return words

def test():
    pass


if __name__ == '__main__':
    """
    to test, run:
        python /home/s1785140/fairseq/fairseq/data/audio/speech_audio_corrector_dataset.py
    """
    test()
