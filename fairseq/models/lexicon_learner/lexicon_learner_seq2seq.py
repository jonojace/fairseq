import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    BaseFairseqModel,
    register_model,
)
from typing import Any, Dict, List, Optional, Tuple
from fairseq.models.lexicon_learner.audio_transformers import (
    TransformerEncoder,
    TransformerDecoder,
)

"""
Commands for debugging training of this model:

MODEL_NAME=debug_seq2seq_ALL
DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
fairseq-train $DATA \
    --tensorboard-logdir tb_logs/$MODEL_NAME \
    --task learn_lexicon \
    --arch lexicon_learner_seq2seq \
    --criterion lexicon_learner \
    --optimizer adam \
    --batch-size 4 \
    --min-train-examples-per-wordtype 10 \
    --max-train-examples-per-wordtype 10 \
    --valid-seen-wordtypes 100 \
    --valid-unseen-wordtypes 100 \
    --valid-examples-per-wordtype 10 \
    --valid-subset valid-seen,valid-unseen \
    --save-dir checkpoints/$MODEL_NAME \
    --save-interval 1 --max-epoch 3 \
    --lr 0.0001 \
    --no-save
"""

@dataclass
class LexiconLearnerSeq2SeqConfig(FairseqDataclass):
    ##############
    # Data
    input_dim: int = field(
        default=1024, metadata={"help": "dimension of input features"}
    )

    ##############
    # Encoder
    enc_num_layers: int = field(
        default=3, metadata={"help": "number of encoder layers"}
    )
    enc_hid_dim: int = field(
        default=128, metadata={"help": "number of hidden dimensions"}
    )
    enc_out_dim: int = field(
        default=128, metadata={"help": "number of output dimensions"}
    )
    enc_dropout_in: float = field(
        default=0.2, metadata={"help": "number of output dimensions"}
    )

    enc_dropout_out: float = field(
        default=0.2, metadata={"help": "number of output dimensions"}
    )

    ##############
    # Decoder


@register_model("lexicon_learner_seq2seq", dataclass=LexiconLearnerSeq2SeqConfig)
class LexiconLearnerSeq2Seq(BaseFairseqModel):
    def __init__(self, cfg: LexiconLearnerSeq2SeqConfig):
        super().__init__()
        self.cfg = cfg

        self.dropout_in = nn.Dropout(p=cfg.enc_dropout_in)
        self.dropout_out = nn.Dropout(p=cfg.enc_dropout_out)

        self.encoder = TransformerEncoder(cfg)
        self.decoder = TransformerDecoder(cfg)

        self.output_projection = nn.Linear(cfg.enc_hid_dim, cfg.enc_out_dim)

        self.summariser = nn.LSTM(
            input_size=cfg.input_dim, #TODO make input dim dynamic depending on detected dimensions
            hidden_size=cfg.enc_hid_dim,
            num_layers=cfg.enc_num_layers,
            dropout=cfg.enc_dropout_out,
            bidirectional=False,
            batch_first=True,
        )


    @classmethod
    def build_model(cls, cfg: LexiconLearnerSeq2SeqConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def forward(self,
                src_tokens,
                src_lengths,
                return_all_hiddens: bool = True,
                features_only: bool = False,
                alignment_layer: Optional[int] = None,
                alignment_heads: Optional[int] = None,
                ):

        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        # NOTE no prev_output_tokens since we do not have ground truth targets for teaching forcing
        # NOTE instead we set it to all 0's, and thus decoder only takes encoder_out as conditioning info
        # NOTE OR 1's where we want to make a prediction, and 0 where we do not want to make a prediction
        # NOTE (i.e. > # of graphemes)
        prev_output_tokens = 0
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            return_all_hiddens=return_all_hiddens,
        )
        summary = self.summariser(decoder_out)

        # decoder_out: tokens that represent pronunciations
        # summary: single timestep summary of pronunciations used to calculate triplet loss
        return decoder_out, summary
