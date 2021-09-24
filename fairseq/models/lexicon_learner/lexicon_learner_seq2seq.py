import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    BaseFairseqModel,
    register_model,
    FairseqEncoder,
)
from typing import Any, Dict, List, Optional, Tuple
import joblib
# from fairseq import utils

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
    input_dim: int = field(
        default=768, metadata={"help": "dimension of input speech features / dim of embedding table for discrete reps such as hubert. Defaults are 1024 for wav2vec2, 768 for hubert."}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_embed_path: str = field(
        default="/home/s1785140/fairseq/examples/lexicon_learner/embeddings/hubert_km100.bin",
        metadata={"help": "filepath to embedding table (i.e. hubert k-means cluster centroid vectors)"}
    )

    ##############
    # Summariser
    summariser_num_layers: int = field(
        default=3, metadata={"help": "number of encoder layers"}
    )
    summariser_hid_dim: int = field(
        default=128, metadata={"help": "number of hidden dimensions"}
    )
    summariser_out_dim: int = field(
        default=128, metadata={"help": "number of output dimensions"}
    )
    summariser_dropout_in: float = field(
        default=0.2, metadata={"help": "number of output dimensions"}
    )
    summariser_dropout_out: float = field(
        default=0.2, metadata={"help": "number of output dimensions"}
    )
    summariser_layer_to_return: int = field(
        default=-1, metadata={"help": "which layer of summariser to use in triplet loss"}
    )

    # Transformer specific
    mask_transformer_outputs: bool = field(
        default=False, metadata={"help": "mask transformer outputs where we do not want to make predictions"}
    )

    # max_source_positions: int = field(
    #     default=100, metadata={"help": "Max source positions. Input is truncated over this length. Use fairseq/examples/lexicon_learner/wordalign_speechreps.py to calculate len of longest speech reps for any wordtype"}
    # )
    # dropout: float = field(
    #     default=0.2, metadata={"help": "Dropout probability"}
    # )
    # attention_dropout: float = field(
    #     default=0.2, metadata={"help": "attention dropout probability "}
    # )
    # no_scale_embedding: bool = field(
    #     default=False, metadata={"help": "if True, dont scale embeddings"}
    # )
    # no_token_positional_embeddings: bool = field(
    #     default=False, metadata={"help": "if set, disables positional embeddings (outside self attention)"}
    # )
    # adaptive_input: bool = field(
    #     default=False, metadata={"help": "variable capacity input representations"}
    # )
    # quant_noise_pq: float = field(
    #     default=0, metadata={"help": "iterative PQ quantization noise at training time"}
    # )
    # # ENCODER
    # encoder_normalize_before: bool = field(
    #     default=False, metadata={"help": "apply layernorm before each encoder block"}
    # )
    # encoder_layerdrop: float = field(
    #     default=0, metadata={"help": "LayerDrop probability for encoder"}
    # )
    # encoder_layers: int = field(
    #     default=3, metadata={"help": "num encoder layers"}
    # )
    # encoder_ffn_embed_dim: int = field(
    #     default=768, metadata={"help": "encoder embedding dimension for FFN"}
    # )
    # encoder_attention_heads: int = field(
    #     default=3, metadata={"help": "num encoder attention heads (multi-headed attention)"}
    # )
    # encoder_learned_pos: bool = field(
    #     default=False, metadata={"help": "use learned positional embeddings in the encoder"}
    # )
    # # DECODER
    # share_decoder_input_output_embed: bool = field(
    #     default=False, metadata={"help": "share decoder input and output embeddings"}
    # )
    # decoder_attention_heads: int = field(
    #     default=3, metadata={"help": "num decoder attention heads (multi-headed attention)"}
    # )
    # decoder_layers: int = field(
    #     default=3, metadata={"help": "num decoder layers"}
    # )
    # decoder_ffn_embed_dim: int = field(
    #     default=768, metadata={"help": "decoder embedding dimension for FFN"}
    # )
    # decoder_layerdrop: float = field(
    #     default=0, metadata={"help": "LayerDrop probability for decoder"}
    # )
    # decoder_learned_pos: bool = field(
    #     default=False, metadata={"help": "use learned positional embeddings in the decoder"}
    # )



class LSTMEncoder(BaseFairseqModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.dropout_in = nn.Dropout(p=cfg.summariser_dropout_in)
        self.dropout_out = nn.Dropout(p=cfg.summariser_dropout_out)

        self.lstm = nn.LSTM(
            input_size=cfg.input_dim, #TODO make input dim dynamic depending on detected dimensions
            hidden_size=cfg.summariser_hid_dim,
            num_layers=cfg.summariser_num_layers,
            dropout=cfg.summariser_dropout_out,
            bidirectional=False,
            batch_first=True,
        )

        self.output_projection = nn.Linear(cfg.summariser_hid_dim, cfg.summariser_out_dim)

    def forward(
            self,
            src_tokens,
            src_lengths,
    ):
        src_tokens = self.dropout_in(src_tokens)

        # Pack the sequence into a PackedSequence object to feed to the LSTM.
        packed_x = nn.utils.rnn.pack_padded_sequence(
            src_tokens,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False, # TODO could have bad side effects? to perf?
        )

        # Get the output from the LSTM.
        _packed_outputs, (final_timestep_hidden, _final_timestep_cell) = self.lstm(packed_x)

        # NB No need to unpack and then pad this packed seq as we simply need "final timestep hidden",
        # We do not need all timesteps which are returned in "_packed_outputs"


        if self.cfg.summariser_num_layers > 1:
            # Only return from one layer of the LSTM
            final_timestep_hidden = final_timestep_hidden.squeeze(0)[self.cfg.summariser_layer_to_return,:,:]
        else:
            final_timestep_hidden = final_timestep_hidden.squeeze(0)


        final_timestep_hidden = self.dropout_out(final_timestep_hidden)

        final_timestep_hidden = self.output_projection(final_timestep_hidden)

        # print("XXX", final_timestep_hidden)

        return {
            # this will have shape `(bsz, hidden_dim)`
            'final_timestep_hidden': final_timestep_hidden,
        }



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
    ):
        super().__init__()
        self.cfg = cfg
        self.task = task

        self.embed_src_tokens = self.build_src_embedding(task.src_dict, cfg.input_dim, cfg.encoder_embed_path)
        self.embed_tgt_tokens = self.build_tgt_embedding(cfg.input_dim)

        # tgt is the target input, in the training you need tgt input as golden truth to do teacher
        # forcing learning, but in the inference, you don’t need tgt, you only need to input the
        # encoded_vector from encoder and the first token in the tgt, usually it’s BOS.
        # TODO add CLA to modify transformer params
        self.transformer = nn.Transformer(
            d_model=cfg.input_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )

        # self.encoder = TransformerEncoder(cfg, task.src_dict, encoder_embed_tokens)
        # self.decoder = TransformerDecoder(cfg, None, decoder_embed_tokens)
        # self.output_projection = nn.Linear(cfg.summariser_hid_dim, cfg.summariser_out_dim)
        self.summariser = LSTMEncoder(cfg)


    @classmethod
    def build_model(cls, cfg: LexiconLearnerSeq2SeqConfig, task=None):
        """Build a new model instance."""

        # encoder_embed_tokens = cls.build_embedding(
        #     task.src_dict, cfg.input_dim, cfg.encoder_embed_path
        # )
        #
        # decoder_embed_tokens = cls.build_existence_embedding(
        #     task.src_dict, cfg.input_dim, cfg.encoder_embed_path
        # )

        return cls(cfg, task)

        # return cls(cfg, task, encoder_embed_tokens, decoder_embed_tokens)

    @classmethod
    def build_src_embedding(cls, dictionary, embed_dim, path):
        """
        load embedding table
        for discretised speech reps such as hubert
        each hubert code looks up a cluster centroid (an embedding)
        """
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        # load blank embedding table
        emb = get_initialised_embedding(num_embeddings, embed_dim, padding_idx)

        # print("xxx",emb)

        # load pretrained embeddings
        kmeans_model = joblib.load(open(path, "rb"))  # this is just a sklearn model
        centroids = torch.Tensor(kmeans_model.cluster_centers_)

        # sanity check
        assert padding_idx == 0 # TODO WARNING HARDCODED
        assert centroids.shape[0] == num_embeddings - 1 # -1 because hubert centroids do not include padding symbol
        assert centroids.shape[1] == embed_dim

        # fill embedding table with pretrained embeddings
        for idx in range(1, len(dictionary)): # iterate over non-padding indices
            emb.weight.data[idx] = centroids[idx-1]

        return emb

    @classmethod
    def build_tgt_embedding(cls, embed_dim):
        """
        create embedding table of two embeddings...
        1's for where we do have audio
        0's or padding vector for where we do not have underlying speech reps
        helps model know when to stop making predictions or predict a stop token
        """
        num_embeddings = 2 # one embedding for padding, one embedding for where we want to make predictions
        padding_idx = 0

        emb = get_initialised_embedding(num_embeddings, embed_dim, padding_idx)

        return emb

    def forward(self,
                src_tokens,
                tgt_tokens, # tokens that indicate how many outputs we wish to output, equal to phones or graphemes for example.
                # src_lengths, # num of speech rep timesteps for each word example
                tgt_lengths, # num of tokens to output per word, equal to phones or graphemes for example.
                # set to None if no hard constraint is desired.
                # return_all_hiddens: bool = True,
                ):

        # encoder_out = self.encoder(
        #     src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        # )
        # # NOTE no prev_output_tokens since we do not have ground truth targets for teaching forcing
        # # NOTE instead we set it to all 0's, and thus decoder only takes encoder_out as conditioning info
        # # NOTE OR 1's where we want to make a prediction, and 0 where we do not want to make a prediction
        # # NOTE (i.e. > # of graphemes)
        # prev_output_tokens = 0
        # decoder_out = self.decoder(
        #     prev_output_tokens,
        #     encoder_out=encoder_out,
        #     src_lengths=src_lengths,
        #     features_only=features_only,
        #     alignment_layer=alignment_layer,
        #     alignment_heads=alignment_heads,
        #     return_all_hiddens=return_all_hiddens,
        # )

        # print(type(src_tokens))
        # print(type(tgt_tokens))
        # print(src_tokens)
        # print(tgt_tokens)

        src = self.embed_src_tokens(src_tokens)
        tgt = self.embed_tgt_tokens(tgt_tokens)
        # tgt = torch.ones(src.size())  # tgt usually is inputs shifted by one-timestep for teacher forcing
        # # however we set it to simpler 0's where we do not want to make a prediction and 1's to where we do
        # # no. of 1's could be the number of phonemes, or graphemes for example

        src_mask = get_mask(src_tokens, padding_idx=0) # mask out src positions where we do not have audio, so we do not attend over those positions
        tgt_mask = get_mask(tgt_tokens, padding_idx=0) # mask out timesteps where we do not want transformer make predictions (note this masks attn...)
        # TODO (is tgt_mask needed???) or should we let model decide when to stop predicting via a stop-token
        memory_mask = src_mask

        continuous_out = self.transformer(
            src=src,
            tgt=tgt,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=memory_mask,
        )

        # mask transformer outputs where we do not want to make predictions
        # so that summariser+triplet loss 100% does not consider them
        if self.cfg.mask_transformer_outputs:
            continuous_out[tgt_mask] = 0.0

        print(continuous_out.size(), tgt_mask.size())
        # print("continuous_out", continuous_out)
        # print("tgt_mask", tgt_mask)
        # print("tgt_lengths", tgt_lengths)

        # TODO discretise the output of transformer?
        # k-means? gumbel softmax?
        discrete_out = continuous_out

        # TODO instead run soft-dtw of continuous_out/discrete_out over anchor,positive,negative sequences
        # TODO so that we get a better loss that is input into triplet loss?
        summary = self.summariser(continuous_out, tgt_lengths)["final_timestep_hidden"]

        net_output = {
            "continuous_out": continuous_out, # represent pronunciations
            "summary": summary, # single timestep summary of pronunciations used to calculate triplet loss
        }
        return net_output

def get_initialised_embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def get_mask(tokens, padding_idx=0):
    """
    return a bool mask for a sequence of tokens
    False for unmasked positions
    True for masked out positions
    """
    return (tokens == padding_idx)
