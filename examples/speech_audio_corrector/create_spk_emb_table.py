"""
create speaker embedding numpy array loaded by get_speaker_embeddings() in text_to_speech.py task

shape of speaker embedding array is [num_spks, embed_dim]
"""

from pathlib import Path
import glob, os
import torch
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", required=True, type=str) # .npy extension
    parser.add_argument("--embeddings_dir", required=True, type=str)
    parser.add_argument("--speaker_list", required=True, type=str)
    args = parser.parse_args()

    with open(args.speaker_list, 'r') as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines]

    for spk in lines:
        print('spk is', spk)

    print("all embeddings in dir:", os.listdir(args.embeddings_dir))

    all_embeddings = []

    for spk in lines:
        spk_emb = torch.load(os.path.join(args.embeddings_dir, f'{spk}.pt')) # [1,1,192] dim

        # squeeze out one singleton dimension [1,1,192] -> [1,192]
        spk_emb = spk_emb.squeeze(0)

        all_embeddings.append(spk_emb)

    print(all_embeddings)

    print(all_embeddings[0].size())

    # concat along given dim to get a single array (i.e. our spk embedding table)
    spk_emb_table = torch.cat(all_embeddings, dim=0)

    print(spk_emb_table.size())

    # verify a row
    row_to_verify = 10
    print('table', spk_emb_table[row_to_verify,:])
    print('original emb', all_embeddings[row_to_verify])

    # save emb table to disk
    # with open(args.outfile, 'w') as f:
    np.save(args.outfile, spk_emb_table.numpy())


if __name__ == "__main__":
    main()
