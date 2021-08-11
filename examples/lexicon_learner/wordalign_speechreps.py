"""
Helper script that takes a folder of speech reps (wav2vec2, mel-spec, etc.)
and aligns them at word-level using MFA alignments.

Speech reps corresponding to word tokens in the corpus are then saved individually to an output folder
with the following structure:
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
    <wordtype>_<utt id>_<numbered occurrence in the utterance>.pt

Example usage:
    cd ~/fairseq
    python examples/lexicon_learner/wordalign_speechreps.py \
        -s /home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/utt_level \
        -a /home/s1785140/data/ljspeech_MFA_alignments \
        -o /home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level
"""

import os
import argparse
import torch
from process_textgrid import process_textgrid
from tqdm import tqdm
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--speech_reps', type=str, required=True,
                    help='path to single non-nested folder containing speech representations (.pt files)')
parser.add_argument('-a', '--alignments', type=str, required=True,
                    help='path to single non-nested folder containing MFA alignments (.TextGrid files)')
parser.add_argument('-o', '--output_directory', type=str, required=True,
                    help='where to write word-level data')
args = parser.parse_args()


def get_wordlevel_reprs(repr, word):
    """
    extract subsequence of 'repr' that corresponds to a particular word
    function expects input to be of dimension 2: (timesteps, hidden_size)
    """

    assert repr.dim() == 2

    start_fraction = word['start'] / word['utt_dur']
    end_fraction = word['end'] / word['utt_dur']
    timesteps = repr.size(0)
    start_idx = round(start_fraction * timesteps)
    end_idx = round(end_fraction * timesteps)

    return repr[start_idx:end_idx, :]

TO_IGNORE = ['SIL', '<unk>']

# sanity check
assert len(os.listdir(args.speech_reps)) == len(os.listdir(args.alignments))

# ensure we always process utts in same alphabetical order
utt_ids = sorted(file.split('.')[0] for file in os.listdir(args.speech_reps))

# split each speech reps file using the word-level alignments
for utt_id in tqdm(utt_ids):
    # load speech reps from disk
    reps = torch.load(f"{args.speech_reps}/{utt_id}.pt")
    reps.requires_grad = False

    # check dimensions and change them if necessary
    if reps.dim() == 3:
        # we assume (batch, timesteps, hidden_size)
        reps = reps.squeeze(0)  # squeeze away batch dimension
    elif reps.dim() == 2:
        pass  # all is good!
    else:
        raise ValueError("speech representations have an incorrect number of dimensions")

    # load alignments from disk
    _phones_align, words_align = process_textgrid(textgrid_path=f"{args.alignments}/{utt_id}.TextGrid")

    # for naming files saved to disk
    # create counter to keep track of the num of times we see a word in an utterance
    word_occ_in_utt_counter = Counter()

    for word_align in words_align:
        # determine if word or silence
        wordtype = word_align['graphemes']
        if wordtype in TO_IGNORE:
            pass
        else:
            # perform word-level splitting of speech reps
            word_occ_in_utt_counter[wordtype] += 1
            word_reps = get_wordlevel_reprs(reps, word_align)
            # create filepath for saving to disk
            # format is: data_path/<wordtype>/<wordtype>_<utt id>_<numbered occurrence in the utterance>.pt
            save_folder = os.path.join(args.output_directory, wordtype)
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f'{wordtype}__{utt_id}__occ{word_occ_in_utt_counter[wordtype]}__len{word_reps.size(0)}.pt')
            # save to disk
            torch.save(word_reps, save_path)

            # FOR DEBUGGING
            # print(wordtype, word_reps.size(), save_path)
