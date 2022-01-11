import torch
from transformers import HubertForCTC, Wav2Vec2Processor
from datasets import load_dataset
import soundfile as sf
import librosa

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

def replace_pad(l, new_symbol='-'):
    """<pad> refers to epsilon in CTC replace with another symbol for readability"""
    new_l = []
    for x in l:
        if x == "<pad>":
            new_l.append(new_symbol)
        else:
            new_l.append(x)
    return new_l


from itertools import groupby


def reduce_tokens(
        tokens,
        pad_symbol="<pad>",
        word_boundary_symbol="|",
        remove_epsilons=True,
        no_repeat_epsilons=False,
        no_repeat_word_boundaries=False,
        no_repeat_graphemes=False,
):
    """
    reduce a sequence of CTC output tokens that contains

    args:
        tokens: list of CTC model tokens to reduce
        remove_epsilons: whether or not to leave epsilons in
        no_repeat_epsilons: whether to reduce repeated epsilons to just one
        no_repeat_graphemes: whether to reduce repeated graphemes to just one
    """
    reduced_tokens = []
    all_symbols = []
    all_durations = []

    for symbol, group in groupby(tokens):
        duration = sum(1 for _ in group)
        all_symbols.append(symbol)
        all_durations.append(duration)

        if symbol == pad_symbol:
            if remove_epsilons:
                pass
            elif no_repeat_epsilons:
                reduced_tokens.append(symbol)
            else:
                reduced_tokens.extend(duration * [symbol])
        elif symbol == word_boundary_symbol:
            if no_repeat_word_boundaries:
                reduced_tokens.append(symbol)
            else:
                reduced_tokens.extend(duration * [symbol])
        else:
            if no_repeat_graphemes:
                reduced_tokens.append(symbol)
            else:
                reduced_tokens.extend(duration * [symbol])

    return reduced_tokens, all_symbols, all_durations


import os
wav_dir = "/home/s1785140/data/LJSpeech-1.1/wavs"
wavs = os.listdir(wav_dir)
wav_paths = [os.path.join(wav_dir, wav) for wav in wavs]

from tqdm import tqdm

outdir = "/home/s1785140/data/LJSpeech-1.1/hubert_asr_transcription"
outdir_with_repeats = "/home/s1785140/data/LJSpeech-1.1/hubert_asr_transcription_with_grapheme_repeats"
outdir_raw_outputs = "/home/s1785140/data/LJSpeech-1.1/hubert_asr_raw_outputs"

os.makedirs(outdir, exist_ok=True)
os.makedirs(outdir_with_repeats, exist_ok=True)
os.makedirs(outdir_raw_outputs, exist_ok=True)

print(len(wav_paths))

all_transcriptions = []
all_alt_transcriptions = []
all_raw_outputs = []

ljspeech_sampling_rate = 22050
hubert_sampling_rate = 16000

for i, wav_path in enumerate(tqdm(wav_paths[:])):
    speech, _ = sf.read(wav_path)
    speech = librosa.resample(speech, ljspeech_sampling_rate, hubert_sampling_rate)
    utt_id = wav_path.split('/')[-1].split('.')[0]
    input_values = processor(speech, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    filtered_tokens = processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist(), skip_special_tokens=False)
    reduced_tokens, all_symbols, all_durations = reduce_tokens(
        filtered_tokens,
        remove_epsilons=True,
        no_repeat_epsilons=False,
        no_repeat_word_boundaries=True,
        no_repeat_graphemes=False
    )
    alt_transcription = replace_pad(reduced_tokens)
    alt_transcription = [sym if sym != "|" else " " for sym in alt_transcription]
    alt_transcription = "".join(alt_transcription)

    raw_outputs = replace_pad(filtered_tokens)
    raw_outputs = [sym if sym != "|" else " " for sym in raw_outputs]
    raw_outputs = "".join(raw_outputs)

    print(i, '===', raw_outputs)
    print(i, '===', transcription)
    print(i, '===', alt_transcription)

    all_transcriptions.append(f"{utt_id}||{transcription.lower()}")
    all_alt_transcriptions.append(f"{utt_id}||{alt_transcription.lower()}")
    all_raw_outputs.append(f"{utt_id}||{raw_outputs.lower()}")

    outfile = f"{utt_id}.txt"

    # save proper transcription
    with open(os.path.join(outdir, outfile), 'w') as f:
        f.write(all_transcriptions[-1])

    # save alternative transcription
    with open(os.path.join(outdir_with_repeats, outfile), 'w') as f:
        f.write(all_alt_transcriptions[-1])

    # save raw outputs
    with open(os.path.join(outdir_raw_outputs, outfile), 'w') as f:
        f.write(all_raw_outputs[-1])

outfile = f"metadata.csv"

all_transcriptions = sorted(all_transcriptions)
all_alt_transcriptions = sorted(all_alt_transcriptions)
all_raw_outputs = sorted(all_raw_outputs)

print(all_transcriptions)
print(all_alt_transcriptions)
print(all_raw_outputs)

# save proper transcription
with open(os.path.join(outdir, outfile), 'w') as f:
    f.write("\n".join(all_transcriptions))

# save alternative transcription
with open(os.path.join(outdir_with_repeats, outfile), 'w') as f:
    f.write("\n".join(all_alt_transcriptions))

# save raw outputs
with open(os.path.join(outdir_raw_outputs, outfile), 'w') as f:
    f.write("\n".join(all_raw_outputs))
