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
from fairseq import utils
import joblib

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
    # TODO remove this as an argument and dynamically set it dependent on the input features automatically?
    encoder_input_dim: int = field(
        default=768, metadata={"help": "dimension of input speech features / dim of embedding table for discrete reps such as hubert. Defaults are 1024 for wav2vec2, 768 for hubert."}
    )
    encoder_embed_path: str = field(
        default="/home/s1785140/fairseq/examples/lexicon_learner/embeddings/hubert_km100.bin",
        metadata={"help": "filepath to embedding table (i.e. hubert k-means cluster centroid vectors)"}
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
    """
    Inputs are discrete tokens that are looked up via an embedding table
    (because default transformers req discrete inputs)

    Outputs are either continous or discrete
    """
    def __init__(
            self,
            cfg: LexiconLearnerSeq2SeqConfig,
            task,
            encoder_embed_tokens,
    ):
        super().__init__()
        self.cfg = cfg
        self.task = task

        self.dropout_in = nn.Dropout(p=cfg.enc_dropout_in)
        self.dropout_out = nn.Dropout(p=cfg.enc_dropout_out)

        # TODO pass a separate transformerencoder/decoder cfg (Namespace) to these object instantiations
        self.encoder = TransformerEncoder(cfg, task.src_dict, encoder_embed_tokens)
        self.decoder = TransformerDecoder(cfg, None, None) # TODO modify TransformerDecoder since no target dict?

        self.output_projection = nn.Linear(cfg.enc_hid_dim, cfg.enc_out_dim)

        self.summariser = nn.LSTM(
            input_size=cfg.encoder_input_dim, #TODO make input dim dynamic depending on detected dimensions
            hidden_size=cfg.enc_hid_dim,
            num_layers=cfg.enc_num_layers,
            dropout=cfg.enc_dropout_out,
            bidirectional=False,
            batch_first=True,
        )


    @classmethod
    def build_model(cls, cfg: LexiconLearnerSeq2SeqConfig, task=None):
        """Build a new model instance."""

        encoder_embed_tokens = cls.build_embedding(
            task.src_dict, cfg.encoder_input_dim, cfg.encoder_embed_path
        )

        return cls(cfg, task, encoder_embed_tokens)

    @classmethod
    def build_embedding(cls, dictionary, embed_dim, path):
        """
        load embedding table
        for discretised speech reps such as hubert
        each hubert code looks up a cluster centroid (an embedding)
        """
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        # load blank embedding table
        emb = Embedding(num_embeddings, embed_dim, padding_idx)

        # load pretrained embeddings
        # key=token, val=embedding
        # kmeans_model_path = '/home/s1785140/fairseq/examples/lexicon_learner/embeddings/hubert_km100.bin'
        kmeans_model = joblib.load(open(path, "rb"))  # this is just a sklearn model
        centroids = torch.Tensor(kmeans_model.cluster_centers_)

        # print(type(centroids))
        # print(len(centroids))
        # print(centroids.shape)
        assert centroids.shape[0] + 1 == num_embeddings # +1 because hubert centroids do not include padding symbol
        assert centroids.shape[1] == embed_dim

        # fill embedding table with pretrained embeddings
        assert padding_idx == 0
        for idx in range(1, len(dictionary)): # iterate over non-padding indices
            emb.weight.data[idx] = centroids[idx-1]

        # embed_dict = utils.parse_embedding(path)
        # utils.load_embedding(embed_dict, dictionary, emb)

        return emb

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

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
