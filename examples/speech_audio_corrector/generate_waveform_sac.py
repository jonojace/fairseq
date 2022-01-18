# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import soundfile as sf
import sys
import torch
import torchaudio
import os

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.tasks.text_to_speech import plot_tts_output
from fairseq.data.audio.text_to_speech_dataset import TextToSpeechDataset


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_parser():
    parser = options.get_speech_generation_parser()
    parser.add_argument("--dump-features", action="store_true")
    parser.add_argument("--dump-waveforms", action="store_true")
    parser.add_argument("--dump-attentions", action="store_true")
    parser.add_argument("--dump-eos-probs", action="store_true")
    parser.add_argument("--dump-plots", action="store_true")
    parser.add_argument("--dump-target", action="store_true")
    parser.add_argument("--output-sample-rate", default=22050, type=int)
    parser.add_argument("--teacher-forcing", action="store_true")
    parser.add_argument(
        "--audio-format", type=str, default="wav", choices=["wav", "flac"]
    )
    parser.add_argument(
        "--txt-file", type=str, default="",
        help="path to txt file of utterances to generate."
    )
    parser.add_argument("--speechreps-add-mask-tokens", action="store_true")
    parser.add_argument("--add-count-to-filename", action="store_true")
    parser.add_argument("--use-external-speechreps", action="store_true",
                        help="Use this flag if you want to use speechreps from the external dataset to do inference.")

    return parser

def sac_friendly_text(words_and_speechreps, incl_codes=False,
                      upper=False, wrap_speechreps_word=True, delimiter=" "):
    """
    given
    (how, None)
    (you, None)
    (doing, HUB2 HUB35...)

    generate a text of following format
    "how are <you-2,35,35,33>"
    """
    rv = []
    for word, speechreps in words_and_speechreps:
        if speechreps is None:
            rv.append(word)
        else:
            s = f"{word.upper() if upper else word}"
            if incl_codes:
                raise NotImplementedError # finish this, this currently breaks soundfile as soundfile can't deal with
                # writing such long filenames to disk
                codes_str = "-".join(str(sr) for sr in speechreps)
                s += f"-{codes_str}"
            if wrap_speechreps_word:
                s = f"<{s}>"
            rv.append(s)

    return delimiter.join(rv)

def strip_pointy_brackets(s):
    return "".join(c for c in s if c not in ["<", ">"])

def postprocess_results(
        dataset: TextToSpeechDataset, sample, hypos, resample_fn, dump_target, sort_by_text=True
):
    def to_np(x):
        return None if x is None else x.detach().cpu().numpy()

    if sample["id"] is not None:
        sample_ids = [dataset.ids[i] for i in sample["id"].tolist()]
    else:
        sample_ids = [None for _ in hypos]

    texts = sample["raw_texts"]
    attns = [to_np(hypo["attn"]) for hypo in hypos]
    eos_probs = [to_np(hypo.get("eos_prob", None)) for hypo in hypos]
    feat_preds = [to_np(hypo["feature"]) for hypo in hypos]
    wave_preds = [to_np(resample_fn(h["waveform"])) for h in hypos]

    if sample["words_and_speechreps"] is not None:
        sac_friendly_texts = [sac_friendly_text(x, incl_codes=False) for x in sample["words_and_speechreps"]]
    else:
        sac_friendly_texts = [None for _ in hypos]

    if dump_target:
        feat_targs = [to_np(hypo["targ_feature"]) for hypo in hypos]
        wave_targs = [to_np(resample_fn(h["targ_waveform"])) for h in hypos]
    else:
        feat_targs = [None for _ in hypos]
        wave_targs = [None for _ in hypos]

    # sort the samples in batch by the text seq
    zipped = list(zip(sample_ids, texts, attns, eos_probs, feat_preds, wave_preds,
               feat_targs, wave_targs, sac_friendly_texts))

    # print("zipped text before sort", [tup[1] for tup in zipped])

    if sort_by_text:
        # strip_pointy_brackets so that the brackets are not considered when sorting
        zipped = sorted(zipped, key=lambda tup: strip_pointy_brackets(tup[1]))
        # print("zipped after sort", zipped)
        # print("zipped after sort", list(zipped))
        # print("zipped text after sort", [tup[1] for tup in zipped])

    return zipped


def dedupe_adjacent(iterable, token_to_dedupe="<mask>"):
    prev = object()
    for item in iterable:
        if item != token_to_dedupe:
            prev = item
            yield item
        elif item != prev: # here item is equal to token_to_dedupe
            prev = item
            yield item

def dump_result(
        is_na_model,
        args,
        count,
        vocoder,
        add_count_to_filename,
        sample_id,
        text,
        attn,
        eos_prob,
        feat_pred,
        wave_pred,
        feat_targ,
        wave_targ,
        sac_friendly_text,
):
    # add useful info to filename
    if sample_id and sac_friendly_text:
        filename_no_ext = f"{sample_id}-{sac_friendly_text}"
    else:
        if add_count_to_filename:
            filename_no_ext = f"{count}-{text}"
        else:
            filename_no_ext = f"{text}"

    sample_rate = args.output_sample_rate
    out_root = Path(args.results_path)
    if args.dump_features:
        feat_dir = out_root / "feat"
        feat_dir.mkdir(exist_ok=True, parents=True)
        np.save(feat_dir / f"{filename_no_ext}.npy", feat_pred)
        if args.dump_target:
            feat_tgt_dir = out_root / "feat_tgt"
            feat_tgt_dir.mkdir(exist_ok=True, parents=True)
            np.save(feat_tgt_dir / f"{filename_no_ext}.npy", feat_targ)
    if args.dump_attentions:
        attn_dir = out_root / "attn"
        attn_dir.mkdir(exist_ok=True, parents=True)
        np.save(attn_dir / f"{filename_no_ext}.npy", attn.numpy())
    if args.dump_eos_probs and not is_na_model:
        eos_dir = out_root / "eos"
        eos_dir.mkdir(exist_ok=True, parents=True)
        np.save(eos_dir / f"{filename_no_ext}.npy", eos_prob)

    if args.dump_plots:
        images = [feat_pred.T] if is_na_model else [feat_pred.T, attn]
        names = ["output"] if is_na_model else ["output", "alignment"]
        if feat_targ is not None:
            images = [feat_targ.T] + images
            names = [f"target (idx={filename_no_ext})"] + names
        if is_na_model:
            plot_tts_output(images, names, attn, "alignment", suptitle=sac_friendly_text)
        else:
            plot_tts_output(images, names, eos_prob, "eos prob", suptitle=sac_friendly_text)
        plot_dir = out_root / "plot"
        plot_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_dir / f"{filename_no_ext}.png")
        plt.close()

    if args.dump_waveforms:
        ext = args.audio_format
        if wave_pred is not None:
            wav_dir = out_root / f"{ext}_{sample_rate}hz_{vocoder}"
            wav_dir.mkdir(exist_ok=True, parents=True)
            sf.write(wav_dir / f"{filename_no_ext}.{ext}", wave_pred, sample_rate)
        if args.dump_target and wave_targ is not None:
            wav_tgt_dir = out_root / f"{ext}_{sample_rate}hz_{vocoder}_tgt"
            wav_tgt_dir.mkdir(exist_ok=True, parents=True)
            sf.write(wav_tgt_dir / f"{filename_no_ext}.{ext}", wave_targ, sample_rate)


def filter_utts_whose_words_do_not_have_speechreps(
        utts,
        dataset,
        use_external_speechreps=False,
        ignore_list=[]
):
    missing_words = set()
    new_utts = []

    # print("DEBUG", list(dataset.word2speechreps.keys()))

    for utt in utts:
        for token in utt.split(" "):
            if token.startswith("<") and token.endswith(">"):
                word = token.lstrip("<").rstrip(">")
            else:
                word = token

            w2sr = dataset.ext_word2speechreps if use_external_speechreps else dataset.word2speechreps

            if word not in w2sr and word not in ignore_list:
                missing_words.add(word)
                break
        else:
            new_utts.append(utt)

    if len(missing_words) > 0:
        print(f"\nWARNING {len(missing_words)} (out of {len(utts)}) utts left out from inference. Words not in dataset.word2speechreps are:", missing_words)

    # print(f"DEBUG", len(utts), len(new_utts))

    return new_utts

def main(args):
    assert(args.dump_features or args.dump_waveforms or args.dump_attentions
           or args.dump_eos_probs or args.dump_plots)
    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 8000
    logger.info(args)

    # setup model and task
    use_cuda = torch.cuda.is_available() and not args.cpu
    task = tasks.setup_task(args)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        task=task,
    )
    model = models[0].cuda() if use_cuda else models[0]

    # use the original n_frames_per_step
    task.args.n_frames_per_step = saved_cfg.task.n_frames_per_step

    # if args.txt_file:
    #     # TODO combine train dev and test so we have more options of word-aligned speech reps to choose from?
    #     task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)
    # else:
    #     task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)

    task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)

    data_cfg = task.data_cfg

    # set resampling function for post processing of model outputs
    sample_rate = data_cfg.config.get("features", {}).get("sample_rate", 22050)
    resample_fn = {
        False: lambda x: x,
        True: lambda x: torchaudio.sox_effects.apply_effects_tensor(
            x.detach().cpu().unsqueeze(0), sample_rate,
            [['rate', str(args.output_sample_rate)]]
        )[0].squeeze(0)
    }.get(args.output_sample_rate != sample_rate)
    if args.output_sample_rate != sample_rate:
        logger.info(f"resampling to {args.output_sample_rate}Hz")

    generator = task.build_generator([model], args)

    dataset = task.dataset(args.gen_subset)

    if args.txt_file:
        # generate test sentences in txt file (WARNING: do not have underlying ground truth audio for obj eval!)
        with open(args.txt_file, 'r') as f:
            test_utts = [l.rstrip("\n") for l in f.readlines() if l != "\n" and not l.startswith("#")]

        print("test_utts", test_utts)
        print("args.use_external_speechreps", args.use_external_speechreps)

        test_utts = filter_utts_whose_words_do_not_have_speechreps(
            test_utts,
            dataset,
            use_external_speechreps=args.use_external_speechreps,
            ignore_list=["how", "is", "pronounced"]
        )
            
        # create mini-batches with given size constraints
        itr = dataset.batch_from_utts(
            test_utts,
            dataset,
            max_sentences=args.batch_size,
            speechreps_add_mask_tokens=args.speechreps_add_mask_tokens,
            use_external_speechreps=args.use_external_speechreps
        )

    else:
        # generate from a subset of corpus (usually test, but can specify train or dev)
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens,
            max_sentences=args.batch_size,
            max_positions=(sys.maxsize, sys.maxsize),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)

    # output to different directory if using external speechreps
    if args.use_external_speechreps:
        args.results_path = os.path.join(args.results_path, 'ext_speechreps')

    Path(args.results_path).mkdir(exist_ok=True, parents=True)

    is_na_model = getattr(model, "NON_AUTOREGRESSIVE", False)

    vocoder = task.args.vocoder
    count = 0
    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            # print("DEBUG", sample["src_texts"])
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            hypos = generator.generate(model, sample, has_targ=args.dump_target)
            for result in postprocess_results(
                    dataset, sample, hypos, resample_fn, args.dump_target,
                    sort_by_text=True if args.txt_file else False,
            ):
                count += 1
                dump_result(is_na_model, args, count, vocoder, args.add_count_to_filename, *result)

    print(f"*** Finished SAC generation of {count} items ***")


def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
