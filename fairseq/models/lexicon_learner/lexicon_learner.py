import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    register_model,
)
from fairseq.modules import (
    TransformerEncoderLayer,
)

"""
Commands for debugging training of this model:

DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
fairseq-train $DATA \
    --task learn_lexicon \
    --arch lexicon_learner \
    --optimizer adam \
    --batch-size 4 \
    --num-wordtypes 100 \
    --max-examples-per-wordtype 100
"""

@dataclass
class LexiconLearnerConfig(FairseqDataclass):
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
    enc_dropout: float = field(
        default=0.2, metadata={"help": "number of output dimensions"}
    )

    ##############
    # Decoder


@register_model("lexicon_learner", dataclass=LexiconLearnerConfig)
class LexiconLearner(BaseFairseqModel):
    def __init__(self, cfg: LexiconLearnerConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = LSTMEncoder(cfg)

    @classmethod
    def build_model(cls, cfg: LexiconLearnerConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def forward(self, x):
        x = self.encoder(x)
        return x


class LSTMEncoder(FairseqEncoder):

    def __init__(self, cfg):
        super().__init__(None)

        self.cfg = cfg

        self.dropout = nn.Dropout(p=cfg.enc_dropout)

        self.lstm = nn.LSTM(
            input_size=cfg.input_dim, #TODO make input dim dynamic depending on detected dimensions
            hidden_size=cfg.enc_hid_dim,
            num_layers=cfg.enc_num_layers,
            bidirectional=False,
            batch_first=True,
        )

        self.output_projection = nn.Linear(cfg.enc_hid_dim, cfg.enc_out_dim)

    def forward(self, src_tokens, src_lengths):

        x = self.dropout(src_tokens)

        # Pack the sequence into a PackedSequence object to feed to the LSTM.
        x = nn.utils.rnn.pack_padded_sequence(src_tokens, src_lengths, batch_first=True)

        # Get the output from the LSTM.
        _outputs, (final_hidden, _final_cell) = self.lstm(x)

        return {
            # this will have shape `(bsz, hidden_dim)`
            'final_hidden': final_hidden.squeeze(0),
        }
