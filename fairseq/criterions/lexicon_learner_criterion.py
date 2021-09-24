import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

@dataclass
class LexiconLearnerCriterionConfig(FairseqDataclass):
    pass

@register_criterion("lexicon_learner", dataclass=LexiconLearnerCriterionConfig)
class LexiconLearnerCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training

        sample is a mini-batch, not a single sample
        """
        # print(sample.keys())

        net_output = model(**sample["net_input"])
        # logits = model.get_logits(net_output).float()
        # target = model.get_targets(sample, net_output)

        # print("YYY", net_output.keys())
        # print("YYY", net_output["final_timestep_hidden"].size())

        loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        # print("YYY", loss)

        # sample_size = sample['sample_size']

        b_sz = net_output["continuous_out"].size(0)
        assert b_sz % 3 == 0
        num_anchors = int(b_sz / 3)
        sample_size = num_anchors

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            # "ntokens": sample["ntokens"],
            # "nsentences": sample["target"].size(0),
            # "sample_size": sample_size,
            # "embeddings": anchors,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce):
        b_sz = net_output["summary"].size(0)
        assert b_sz % 3 == 0
        num_anchors = int(b_sz / 3)
        # decompose net_output["summary"] into anchors, positives, and negatives
        anchor_summaries = net_output["summary"][:num_anchors]
        positive_summaries = net_output["summary"][num_anchors:2*num_anchors]
        negative_summaries = net_output["summary"][2*num_anchors:]

        # print("anchor", anchor)
        #
        # print("positive", positive)
        #
        # print("negative", negative)

        loss = F.triplet_margin_loss(
            anchor_summaries,
            positive_summaries,
            negative_summaries,
            reduction="sum" if reduce else "none",
        )

        # print("loss", loss)

        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        ### Training and validation metrics
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # print("inside reduce_metrics()", loss_sum, sample_size)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            # "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
            # "loss", loss_sum / sample_size, sample_size, round=3,
            "loss", loss_sum / sample_size, sample_size
        )
        metrics.log_scalar("sample_size", sample_size)

        # ### Validation only metrics
        # wordtypes = [log.get("wordtypes", 0) for log in logging_outputs]
        # if wordtypes[0] == 0:
        #     # training
        #     pass
        # else:
        #     # valid_step()
        #     print("AAA", "len(logging_outputs)", len(logging_outputs))
        #     print("AAA", "wordtypes", wordtypes)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
