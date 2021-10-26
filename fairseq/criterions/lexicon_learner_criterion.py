import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions.soft_dtw_cuda import SoftDTW

def partition_batch(net_output, key, num_anchors):
    """partition batch of anchors positives negatives"""
    anchors = net_output[key][:num_anchors]
    positives = net_output[key][num_anchors:2 * num_anchors]
    negatives = net_output[key][2 * num_anchors:]
    return anchors, positives, negatives

@dataclass
class LexiconLearnerCriterionConfig(FairseqDataclass):
    softdtw_gamma: Optional[float] = field(
        default=0.1, metadata={"help": "gamma value for softdtw"}
    )
    reduction_strategy: Optional[str] = field(
        default="mean", metadata={"help": "reduction strategy to handle summed soft dtw losses"}
    )
    triplet_loss_margin: Optional[float] = field(
        default=1.0, metadata={"help": "triplet loss margin"}
    )

@register_criterion("lexicon_learner", dataclass=LexiconLearnerCriterionConfig)
class LexiconLearnerCriterion(FairseqCriterion):
    def __init__(self, task, cfg: LexiconLearnerCriterionConfig):
        super().__init__(task)
        self.cfg = cfg
        self.sdtw = SoftDTW(use_cuda=torch.cuda.is_available(), gamma=cfg.softdtw_gamma)


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

        losses = self.compute_loss(model, net_output, sample, reduce=reduce)

        # print("YYY", loss)

        # sample_size = sample['sample_size']

        b_sz = net_output["continuous_out"].size(0)
        assert b_sz % 3 == 0
        num_anchors = int(b_sz / 3)
        sample_size = num_anchors

        logging_output = {
            "triplet_loss": losses["triplet_loss"].data,
            "sample_size": sample_size,
            # "ntokens": sample["ntokens"],
            # "nsentences": sample["target"].size(0),
            # "sample_size": sample_size,
            # "embeddings": anchors,
        }

        if "pos_loss" in losses:
            logging_output["pos_loss"] = losses["pos_loss"].data

        if "neg_loss" in losses:
            logging_output["neg_loss"] = losses["neg_loss"].data

        return losses["triplet_loss"], sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce):
        device = net_output["continuous_out"].device
        b_sz = net_output["continuous_out"].size(0)
        assert b_sz % 3 == 0
        num_anchors = int(b_sz / 3)

        if model.cfg.sequence_loss_method == "softdtw":
            anchor_outs, positive_outs, negative_outs = partition_batch(net_output, "continuous_out", num_anchors)
            anchor_tgt_lens, positive_tgt_lens, negative_tgt_lens = partition_batch(sample["net_input"], "tgt_lengths", num_anchors)

            # since softdtw loss fn needs x and y to be same length, we iterate over each triplet and truncate away padding.
            losses = []
            pos_losses = []
            neg_losses = []
            for anchor_out, anchor_tgt_len, positive_out, positive_tgt_len, negative_out, negative_tgt_len in zip(anchor_outs, anchor_tgt_lens, positive_outs, positive_tgt_lens, negative_outs, negative_tgt_lens):
                # introduce batch dimension as softdtw loss fn needs it (batchsize, seq_len, dim)
                anchor_out, positive_out, negative_out = anchor_out.unsqueeze(0), positive_out.unsqueeze(0), negative_out.unsqueeze(0)
                # print(anchor_out.size(), anchor_tgt_len)
                # get unpadded sequences
                a = anchor_out[:, :anchor_tgt_len]
                p = positive_out[:, :positive_tgt_len]
                n = negative_out[:, :negative_tgt_len]

                # print(anchor_out.size(), anchor_tgt_len, a.size())

                # get losses
                positive_sdtw_loss = self.sdtw(a, p)
                negative_sdtw_loss = self.sdtw(a, n)
                # print("positive_sdtw_loss", positive_sdtw_loss, "negative_sdtw_loss", negative_sdtw_loss)
                # print("positive_sdtw_loss.requires_grad", positive_sdtw_loss.requires_grad, "negative_sdtw_loss.requires_grad", negative_sdtw_loss.requires_grad)
                # print("(positive_sdtw_loss - negative_sdtw_loss + self.cfg.triplet_loss_margin).requires_grad",(positive_sdtw_loss - negative_sdtw_loss + self.cfg.triplet_loss_margin).requires_grad)
                # get triplet loss
                triplet_loss = torch.max(positive_sdtw_loss - negative_sdtw_loss + self.cfg.triplet_loss_margin, torch.tensor([0.], requires_grad=True, device=device))
                # print("triplet_loss",triplet_loss)
                # print("triplet_loss.requires_grad", triplet_loss.requires_grad)
                losses.append(triplet_loss)
                pos_losses.append(positive_sdtw_loss)
                neg_losses.append(negative_sdtw_loss)

            losses = torch.stack(losses)
            pos_losses = torch.stack(pos_losses)
            neg_losses = torch.stack(neg_losses)

            # print("losses.requires_grad", losses.requires_grad)

            if reduce:
                # print("REDUCING!!!")
                if self.cfg.reduction_strategy == "mean":
                    loss = losses.mean()
                    pos_loss = pos_losses.mean()
                    neg_loss = neg_losses.mean()
                elif self.cfg.reduction_strategy == "sum":
                    loss = losses.sum()
                    pos_loss = pos_losses.sum()
                    neg_loss = neg_losses.sum()
                else:
                    raise ValueError
            else:
                # print("NO REDUCE...")
                loss = losses
                pos_loss = pos_losses
                neg_loss = neg_losses

            # print("loss.requires_grad", loss.requires_grad)

            return {
                "triplet_loss": loss,
                "pos_loss": pos_loss,
                "neg_loss": neg_loss,
            }

        elif model.cfg.sequence_loss_method == "summariser":
            anchor_summaries, positive_summaries, negative_summaries = partition_batch(net_output, "summary", num_anchors)
            loss = F.triplet_margin_loss(
                anchor_summaries,
                positive_summaries,
                negative_summaries,
                reduction="sum" if reduce else "none", # TODO should this be sum or mean???
            )
            loss /= num_anchors # average the loss across samples in the batch so we don't need to change the learning rate when we change the batch size
            return {"triplet_loss": loss}
        else:
            raise ValueError



    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        ### Training and validation metrics
        loss_sum = sum(log.get("triplet_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # print("inside reduce_metrics()", loss_sum, sample_size)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            # "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
            # "loss", loss_sum / sample_size, sample_size, round=3,
            "loss", loss_sum / sample_size, sample_size
        )
        metrics.log_scalar("sample_size", sample_size)

        ### soft-dtw loss specific components
        pos_loss_sum = sum(log.get("pos_loss", 0) for log in logging_outputs)
        neg_loss_sum = sum(log.get("neg_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "pos_loss", pos_loss_sum / sample_size, sample_size
        )
        metrics.log_scalar(
            "neg_loss", neg_loss_sum / sample_size, sample_size
        )

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
