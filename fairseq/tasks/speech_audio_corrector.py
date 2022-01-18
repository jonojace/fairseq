# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import os.path as op

import torch
import torch.nn.functional as F
import numpy as np

from fairseq.data.audio.speech_audio_corrector_dataset import SpeechAudioCorrectorDatasetCreator
from fairseq.tasks import register_task
from fairseq.tasks.text_to_speech import TextToSpeechTask
from fairseq.speech_generator import (
    AutoRegressiveSpeechGenerator, NonAutoregressiveSpeechGenerator,
    TeacherForcingAutoRegressiveSpeechGenerator, SACAutoRegressiveSpeechGenerator
)
from fairseq.data import Dictionary
from pathlib import Path
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
)

logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO
)
logger = logging.getLogger(__name__)


try:
    from tensorboardX import SummaryWriter
except ImportError:
    logger.info("Please install tensorboardX: pip install tensorboardX")
    SummaryWriter = None

#TODO add args from CLA
#TODO randomise_examples


@register_task('speech_audio_corrector')
class SpeechAudioCorrectorTask(TextToSpeechTask):
    @classmethod
    def add_args(cls, parser):
        super(SpeechAudioCorrectorTask, cls).add_args(parser)
        parser.add_argument("--randomise-examples", action="store_true",)
        # parser.add_argument("--incl-external-speechreps", action="store_true",)

    def __init__(self, args, src_dict):
        super().__init__(args, src_dict)

        self.args = args

        # add symbols for SAC to dictionary
        self.src_dict.add_symbol("<mask>")
        K=100
        for i in range(K):
            # add to src_dict entries for hubert codes i.e. HUB0, HUB1, ..., HUB<k-1>
            self.src_dict.add_symbol(f"HUB{i}")

        # print entire dictionary should be graphemes + hubert codes
        print("(symbol, index) mapping in dictionary for Speech Audio Corrector Training:")
        print([(symbol, src_dict.index(symbol)) for symbol in src_dict.symbols])

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith('train')
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)

        self.datasets[split] = SpeechAudioCorrectorDatasetCreator.from_tsv(
            self.args,
            self.args.data, self.data_cfg, split, self.src_dict,
            pre_tokenizer, bpe_tokenizer, is_train_split=is_train_split,
            epoch=epoch, seed=self.args.seed,
            n_frames_per_step=self.args.n_frames_per_step,
            speaker_to_id=self.speaker_to_id,
        )

    def build_generator(self, models, cfg, vocoder=None, **unused):
        if vocoder is None:
            vocoder = self.build_default_vocoder()
        model = models[0]
        if getattr(model, "NON_AUTOREGRESSIVE", False):
            return NonAutoregressiveSpeechGenerator(
                model, vocoder, self.data_cfg
            )
        else:
            generator = SACAutoRegressiveSpeechGenerator
            if getattr(cfg, "teacher_forcing", False):
                generator = TeacherForcingAutoRegressiveSpeechGenerator
                logger.info("Teacher forcing mode for generation")
            return generator(
                model, vocoder, self.data_cfg,
                max_iter=self.args.max_target_positions,
                eos_prob_threshold=self.args.eos_prob_threshold
            )

