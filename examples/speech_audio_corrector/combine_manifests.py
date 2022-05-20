"""
Usage:

python combine_manifests.py \
    --in_tsvs /home/s1785140/data/LJSpeech-1.1/feature_manifest/train.tsv /home/s1785140/data/VCTK_fairseq/feature_manifest/train.tsv \
    --out_tsv /home/s1785140/data/LJ_and_VCTK_fairseq/feature_manifest/train.tsv
"""

import argparse

def check_headers(tsvs):
    content = load_tsv(tsvs[0])
    header = content[0]
    for tsv in tsvs[1:]:
        content = load_tsv(tsv)
        assert header == content[0]

def load_tsv(tsv):
    with open(tsv, 'r') as f:
        lines = f.readlines()
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_tsvs", required=True, nargs='+', default=[], help='list of tsvs to be combined')
    parser.add_argument("--out_tsv", required=True, type=str, help='output tsv')
    args = parser.parse_args()

    # check that headers are same for all input tsvs
    check_headers(args.in_tsvs)

    # get header from first tsv
    contents = [load_tsv(args.in_tsvs[0])[0]]

    # load each of in tsv
    for in_tsv in args.in_tsvs:
        contents.extend(load_tsv(in_tsv)[1:]) # [1:] in order to exclude header from each tsv file

    # save to out tsv
    with open(args.out_tsv, 'w') as f:
        f.writelines(contents)

main()
