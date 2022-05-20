'''
Script to reorganise VCTK corpus for the format expected for the Montreal Forced Aligner

First run this script to reorganise vctk 

Then run mfa as follows:
mfa align /Users/jonojace/data/vctk_for_montreal /Users/jonojace/montreal_forced_aligner/english.dict /Users/jonojace/montreal_forced_aligner/english.zip  /Users/jonojace/data/vctk_montreal_alignments
'''

import os
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True,
                    help='input VCTK folder, i.e. /home/s1785140/data/VCTK-Corpus')
parser.add_argument('--output_dir', type=str, required=True,
                    help='output VCTK folder, i.e. /home/s1785140/data/vctk_for_montreal')

args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

txt_dir = os.path.join(input_dir, "txt")
wav_dir = os.path.join(input_dir, "wav48_silence_trimmed")

def move_file(path, new_extension=""):
    # move into folder in output dir labeled with speaker name
    speaker = path.name.split('_')[0]
    speaker_output_dir = os.path.join(output_dir, speaker)

    if not os.path.isdir(speaker_output_dir):
        os.makedirs(speaker_output_dir, exist_ok=True)
    
    # optionally change extension of file
    if new_extension:
        output_file = path.name.rsplit('.')[0] + new_extension
    else:
        output_file = path.name

    copyfile(path, os.path.join(speaker_output_dir, output_file))

# glob paths for all wav and txt files
for path in tqdm(Path(wav_dir).rglob('*.wav')):
    move_file(path, new_extension='.wav')

for path in tqdm(Path(txt_dir).rglob('*.txt')):
    move_file(path, new_extension='.lab')

