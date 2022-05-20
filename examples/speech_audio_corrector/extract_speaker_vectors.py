from pathlib import Path
import glob, os
import torch
from tqdm import tqdm

import torchaudio
from speechbrain.pretrained import EncoderClassifier

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", required=True, type=str)
args = parser.parse_args()


classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
VEC_DIM = [1,1,192] # dim of ecapa vector outputs
OUTDIR = "/home/s1785140/fairseq/examples/speech_audio_corrector/speaker_embeddings"

def save_speaker_embedding(wav_folder, savename, outdir=Path(OUTDIR), use_tqdm=True):
    wav_paths = glob.glob(os.path.join(wav_folder, '*.wav'))
    summed = torch.zeros(VEC_DIM, dtype=torch.float32)
    if use_tqdm:
        iterator = tqdm
    else:
        iterator = list
    for wav_path in iterator(wav_paths):
        signal, fs = torchaudio.load(wav_path)
        embeddings = classifier.encode_batch(signal)
        summed = summed + embeddings
    avg_embedding = summed / len(wav_paths)
    torch.save(avg_embedding, outdir / f'{savename}.pt')

if args.dataset == "vctk":
    #VCTK
    wav_folder = Path('/home/s1785140/data/VCTK_fairseq/audio_data/VCTK-Corpus-0.92/wav48_silence_trimmed')
    speakers = os.listdir(wav_folder)
    speakers = [s for s in speakers if s != 'log.txt']

    for speaker in tqdm(speakers):
        save_speaker_embedding(wav_folder / speaker, speaker, use_tqdm=False)

if args.dataset == "ljspeech":
    #LJSpeech
    wav_folder = Path('/home/s1785140/data/LJSpeech-1.1/audio_data/LJSpeech-1.1/wavs')

    save_speaker_embedding(wav_folder, 'ljspeech', use_tqdm=True)
