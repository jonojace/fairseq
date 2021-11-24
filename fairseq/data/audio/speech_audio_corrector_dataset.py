from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

import numpy as np
import torch

from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset, SpeechToTextDatasetCreator, S2TDataConfig,
    _collate_frames, get_features_or_waveform
)
from fairseq.data import Dictionary, data_utils as fairseq_data_utils

from fairseq.data.audio.text_to_speech_dataset import (
    TextToSpeechDatasetItem,
    TextToSpeechDataset,
    TextToSpeechDatasetCreator,
)

from fairseq.data.audio.speech_audio_corrector_utils import (
    get_mfa_text,
    get_word_pos,
    get_word2speechreps,
    get_speechreps_for_utt,
    prepend_speechreps_for_dict_encoding,
    two_random_partitions,
    mask_according_to_word_pos,
)

import textgrid
from collections import Counter
from fairseq.tokenizer import tokenize_line

@dataclass
class SpeechAudioCorrectorDatasetItem(TextToSpeechDatasetItem):
    word_pos: Optional[List[int]] = None
    text_len: int = -1
    speechreps_len: int = -1
    segment: torch.Tensor = None

class SpeechAudioCorrectorDataset(TextToSpeechDataset):
    def __init__(
            self,
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
            ids2speechreps: Dict[Any, List[int]] = None, # TODO change the type hint to correct one
            ids2word_alignments: Dict[Any, List[Dict[str, Any]]] = None, # TODO change the type hint to correct one
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

        ################################################################################################################
        # initialise standard TTS dataset
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
        # add SAC specific data structure to this dataset object
        self.ids2word_alignments = ids2word_alignments
        self.word2speechreps = get_word2speechreps(ids, ids2speechreps, ids2word_alignments)
        # TODO optionally save/load this data structure to disk, so can avoid this slow preprocessing step

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

        # tokenize str of text to sequence of graphemes
        text_tokenized = self.tokenize(self.pre_tokenizer, mfa_text)
        text_tokenized = self.tokenize(self.bpe_tokenizer, text_tokenized)

        # split text by whitespace (whitespace in tts (BPE) text is denoted by a special underscore "▁"
        bpe_token_delimiter = " "
        text_tokenized = text_tokenized.split(bpe_token_delimiter)

        # append eos symbol (needed as we will concatenate speech reps to end of this text seq)
        eos_symbol = "</s>"
        text_tokenized.append(eos_symbol)

        # get the word position of each word and symbol in the input text sequence
        word_and_word_pos, word_pos_of_graphemes = get_word_pos(text_tokenized,
                                                                bpe_whitespace_tok="▁",
                                                                boundary_same_pos=True,
                                                                append_eos=False, boundary_pos=0)
        #####################################################################[###########################################
        # Speech reps
        # retrieve speech reps for each word in mfa text, inserting a <sep> token in between each word-aligned speech rep chunk
        speechreps, word_pos_of_speechreps = get_speechreps_for_utt(
            word_and_word_pos, utt_id, self.word2speechreps,
            randomise_examples=False, remove_dup_prob=1.0,
            remove_dup_rand_num=False, dropout_p=0.0,
            append_eos=True, eos_symbol="</s>",
        )
        speechreps = prepend_speechreps_for_dict_encoding(speechreps, prepend_str="HUB",
                                                          ignore_eos=True, eos_symbol="</s>")
        ################################################################################################################
        # Perform complimentary masking at the word level
        word_positions = [word_pos for wordtype, word_pos in word_and_word_pos]
        text_mask_word_positions, speechreps_mask_word_positions = two_random_partitions(word_positions, p=0.5)
        text_tokenized_masked = mask_according_to_word_pos(text_tokenized, word_pos_of_graphemes,
                                                           text_mask_word_positions,
                                                           mask_token="<mask>")
        speechreps_masked = mask_according_to_word_pos(speechreps, word_pos_of_speechreps,
                                                           speechreps_mask_word_positions,
                                                           mask_token="<mask>")
        ################################################################################################################
        # Encode graphemes and speech reps
        # encode text into int indices, one for each grapheme (incl. whitespace)
        target = self.tgt_dict.encode_line(
                bpe_token_delimiter.join(text_tokenized_masked), add_if_not_exist=False, append_eos=False
        ).long()
        assert len(word_pos_of_graphemes) == len(target), f"{len(word_pos_of_graphemes)} != {len(target)}"

        # encode hubert codes into int indices, one for each hubert code
        encoded_speechreps = self.tgt_dict.encode_line(
                bpe_token_delimiter.join(speechreps_masked), add_if_not_exist=False, append_eos=False
        ).long()
        assert len(word_pos_of_speechreps) == len(encoded_speechreps), f"{len(word_pos_of_speechreps)} != {len(encoded_speechreps)}"
        ################################################################################################################
        # Get length information for text and speechreps
        text_len = len(target)
        speechreps_len = len(encoded_speechreps)
        ################################################################################################################
        # Concat encoded text to encoded hubert codes for Speech Audio Corrector transformer inputs
        target = torch.cat([target, encoded_speechreps])
        word_pos_all_timesteps = torch.LongTensor(word_pos_of_graphemes + word_pos_of_speechreps)
        ################################################################################################################
        # Create segment information tensor (0 is text, 1 is speechrep)
        segment = torch.zeros(target.size()).long()
        segment[text_len:] = 1

        return SpeechAudioCorrectorDatasetItem(
            index=index, source=t2s.source,
            target=target, speaker_id=t2s.speaker_id,
            word_pos=word_pos_all_timesteps,
            text_len=text_len, speechreps_len=speechreps_len,
            segment=segment,
        )

    def collater(self, samples: List[SpeechAudioCorrectorDatasetItem]) -> Dict[str, Any]:
        if len(samples) == 0:
            return {}

        print(f"len(samples)/batch_sz == {len(samples)}")

        ################################################################################################################
        # Audio outputs

        # TODO should we be sorting by target length instead???
        # sort items in batch by src length
        src_lengths, order = torch.tensor(
            [s.target.shape[0] for s in samples], dtype=torch.long
        ).sort(descending=True)

        # collate audio (which is targets in TTS)
        # _collate_frames performs zero padding to len of longest audio
        audio_feat = _collate_frames(
            [s.source for s in samples], self.cfg.use_audio_input
        ).index_select(0, order)

        # get audio original lens before zero padding
        target_lengths = torch.tensor(
            [s.source.shape[0] for s in samples], dtype=torch.long
        ).index_select(0, order)

        ################################################################################################################
        # Text and speechreps inputs

        # inputs (graphemes concatenated with speechreps for speech audio corrector training)
        src_tokens = fairseq_data_utils.collate_tokens(
            [s.target for s in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        ).index_select(0, order)

        # word positions

        # segments

        ################################################################################################################
        # Misc.

        # speaker id
        speaker = None
        if self.speaker_to_id is not None:
            speaker = torch.tensor(
                [s.speaker_id for s in samples], dtype=torch.long
            ).index_select(0, order).view(-1, 1)

        # get time-shifted audio for teacher forced training of TTS decoder
        bsz, _, d = audio_feat.size()
        prev_output_tokens = torch.cat(
            (audio_feat.new_zeros((bsz, 1, d)), audio_feat[:, :-1, :]), dim=1
        )

        # get human readable version of TTS inputs
        src_texts = [self.tgt_dict.string(samples[i].target) for i in order]

        # index of utt in the corpus
        id_ = torch.tensor([s.index for s in samples],
                           dtype=torch.long).index_select(0, order)

        return {
            "id": id_,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "prev_output_tokens": prev_output_tokens,
            },
            "speaker": speaker,
            "target": audio_feat,
            "target_lengths": target_lengths,
            "ntokens": sum(target_lengths).item(),
            "nsentences": len(samples),
            "src_texts": src_texts,
        }

class SpeechAudioCorrectorDatasetCreator(TextToSpeechDatasetCreator):
    @classmethod
    def _from_list(
        cls,
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
        # - speechrep codes for each utt
        # - word alignments for each utt
        speechrep_file = "/home/s1785140/fairseq/examples/speech_audio_corrector/lj_speech_quantized.txt"
        alignments_dir = "/home/s1785140/data/ljspeech_MFA_alignments_from_fb"
        ids2speechreps = cls.load_speechreps(speechrep_file)
        ids2word_alignments = cls.load_word_alignments(ids, alignments_dir)

        sac_dataset = SpeechAudioCorrectorDataset(
            split_name, is_train_split, cfg, audio_paths, n_frames,
            src_texts=src_texts, tgt_texts=tgt_texts, speakers=speakers,
            src_langs=src_langs, tgt_langs=tgt_langs, ids=ids,
            tgt_dict=tgt_dict, pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer, n_frames_per_step=n_frames_per_step,
            speaker_to_id=speaker_to_id,
            durations=durations, pitches=pitches, energies=energies,
            ids2speechreps=ids2speechreps, ids2word_alignments=ids2word_alignments,
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
