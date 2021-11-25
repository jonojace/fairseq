# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
from typing import Any, Dict, List
from functools import lru_cache
from dataclasses import dataclass, field

import torch
from omegaconf import II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import lengths_to_mask
import torch.nn.functional as F

from fairseq.criterions.tacotron2_loss import (
    Tacotron2CriterionConfig,
    Tacotron2Criterion,
    GuidedAttentionLoss,
)

logger = logging.getLogger(__name__)


@dataclass
class SACCriterionConfig(Tacotron2CriterionConfig):
    pass

@register_criterion("sac_tts", dataclass=SACCriterionConfig)
class SACCriterion(Tacotron2Criterion):
    def forward(self, model, sample, reduction="mean"):
        bsz, max_len, _ = sample["target"].size()
        feat_tgt = sample["target"]
        feat_len = sample["target_lengths"].view(bsz, 1).expand(-1, max_len)
        eos_tgt = torch.arange(max_len).to(sample["target"].device)
        eos_tgt = eos_tgt.view(1, max_len).expand(bsz, -1)
        eos_tgt = (eos_tgt == (feat_len - 1)).float()
        src_tokens = sample["net_input"]["src_tokens"]
        src_word_pos = sample["net_input"]["src_word_pos"]
        src_segments = sample["net_input"]["src_segments"]
        src_lens = sample["net_input"]["src_lengths"]
        tgt_lens = sample["target_lengths"]

        feat_out, eos_out, extra = model(
            src_tokens=src_tokens,
            src_word_pos=src_word_pos,
            src_segments=src_segments,
            src_lengths=src_lens,
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            incremental_state=None,
            target_lengths=tgt_lens,
            speaker=sample["speaker"]
        )

        l1_loss, mse_loss, eos_loss = self.compute_loss(
            extra["feature_out"], feat_out, eos_out, feat_tgt, eos_tgt,
            tgt_lens, reduction,
        )
        attn_loss = torch.tensor(0.).type_as(l1_loss)
        if self.guided_attn is not None:
            attn_loss = self.guided_attn(extra['attn'], src_lens, tgt_lens, reduction)
        ctc_loss = torch.tensor(0.).type_as(l1_loss)
        if self.ctc_weight > 0.:
            net_output = (feat_out, eos_out, extra)
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.transpose(0, 1)  # T x B x C
            src_mask = lengths_to_mask(src_lens)
            src_tokens_flat = src_tokens.masked_select(src_mask)
            ctc_loss = F.ctc_loss(
                lprobs, src_tokens_flat, tgt_lens, src_lens,
                reduction=reduction, zero_infinity=True
            ) * self.ctc_weight
        loss = l1_loss + mse_loss + eos_loss + attn_loss + ctc_loss

        sample_size = sample["nsentences"] if self.sentence_avg \
            else sample["ntokens"]
        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "l1_loss": utils.item(l1_loss.data),
            "mse_loss": utils.item(mse_loss.data),
            "eos_loss": utils.item(eos_loss.data),
            "attn_loss": utils.item(attn_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
        }
        return loss, sample_size, logging_output
