# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional

import torch
from torch import nn

from fairseq.models import (FairseqEncoder, FairseqEncoderDecoderModel,
                            FairseqIncrementalDecoder, register_model,
                            register_model_architecture)
from fairseq.modules import (
    TransformerEncoderLayer, TransformerDecoderLayer
)
from fairseq.models.text_to_speech.tacotron2 import Prenet, Postnet
from fairseq.modules import LayerNorm, PositionalEmbedding, FairseqDropout
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq import utils

logger = logging.getLogger(__name__)


def encoder_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


def Embedding(num_embeddings, embedding_dim):
    m = nn.Embedding(num_embeddings, embedding_dim)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m


class SACTransformerEncoder(FairseqEncoder):
    def __init__(self, args, src_dict, embed_speaker):
        super().__init__(src_dict)
        self.padding_idx = src_dict.pad()
        self.token_pos_pad_idx = 0
        self.word_pos_pad_idx = 0 # 0: pad, 1 2 3 4 5 ...: word positions
        self.segment_pad_idx = 0 # 0: pad, 1: graphemes, 2: speechreps
        self.embed_speaker = embed_speaker
        self.spk_emb_proj = None
        if embed_speaker is not None:
            self.spk_emb_proj = nn.Linear(
                args.encoder_embed_dim + args.speaker_embed_dim,
                args.encoder_embed_dim
            )

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim,
                                         padding_idx=self.padding_idx)
        num_segments = 2 # grapheme and speechreps segments
        self.embed_segments = nn.Embedding(num_segments + 1, # +1 for the padding symbol
                                           args.encoder_embed_dim,
                                           padding_idx=self.segment_pad_idx)
        assert(args.encoder_conv_kernel_size % 2 == 1)
        self.prenet = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(args.encoder_embed_dim, args.encoder_embed_dim,
                          kernel_size=args.encoder_conv_kernel_size,
                          padding=((args.encoder_conv_kernel_size - 1) // 2)),
                nn.BatchNorm1d(args.encoder_embed_dim),
                nn.ReLU(),
                nn.Dropout(args.encoder_dropout),
            )
            for _ in range(args.encoder_conv_layers)
        )
        self.prenet_proj = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )

        # Need to supply padding_idx when creating positional embedding
        # Padding timesteps are those beyond length of audio (upto max seq len in batch) in order to form batches of inputs
        # if fairseq/fairseq/modules/sinusoidal_positional_embedding.py receives padding_idx
        # it sets the embedding corresponding to padding_idx to 0. This is desirable
        # as we do not want the model to think that timesteps corresponding to padding are meaningful.
        assert self.token_pos_pad_idx == self.word_pos_pad_idx
        self.embed_positions = PositionalEmbedding(
            num_embeddings=args.max_source_positions,
            embedding_dim=args.encoder_embed_dim,
            padding_idx=self.word_pos_pad_idx,
            # padding_idx=self.padding_idx,
            learned=False
        )

        self.no_word_pos = args.no_word_pos

        self.seg_emb_alpha = nn.Parameter(torch.ones(1))
        self.pos_emb_alpha = nn.Parameter(torch.ones(1))
        if not self.no_word_pos:
            self.word_pos_emb_alpha = nn.Parameter(torch.ones(1))
            self.morph_word_pos = nn.Linear(
                args.encoder_embed_dim, args.encoder_embed_dim
            )

        self.transformer_layers = nn.ModuleList(
            TransformerEncoderLayer(args)
            for _ in range(args.encoder_transformer_layers)
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self.dont_mask_encoder_out_speech_timesteps = args.dont_mask_encoder_out_speech_timesteps

        self.apply(encoder_init)

    def forward(self, src_tokens, src_token_pos, src_word_pos, src_segments, src_lengths=None, speaker=None, **kwargs):
        ################################################################################################################
        # Create inputs

        # print("in forward() of sac_transformer")
        # print("b_sz", src_tokens.size(0))

        # graphemes+speechreps
        x = self.embed_tokens(src_tokens)
        x = x.transpose(1, 2).contiguous()  # B x T x C -> B x C x T
        for conv in self.prenet:
            x = conv(x)
        x = x.transpose(1, 2).contiguous()  # B x C x T -> B x T x C
        x = self.prenet_proj(x)

        # segment information (text or speechreps segment)
        embedded_src_segments = self.embed_segments(src_segments)
        x += self.seg_emb_alpha * embedded_src_segments

        # token positions
        embedded_token_positions = self.embed_positions(
            input=src_token_pos,
            positions=src_token_pos,
        )
        x += self.pos_emb_alpha * embedded_token_positions

        #############################################################################################
        # print("src_segments[0]", src_segments[0])
        # print("embedded_src_segments[0].size()", embedded_src_segments[0].size())
        # print("embedded_src_segments[0].sum(dim=1)", embedded_src_segments[0].sum(dim=1))

        # print("src_segments[1]", src_segments[1])
        # print("embedded_src_segments[1].size()", embedded_src_segments[1].size())
        # print("embedded_src_segments[1].sum(dim=1)", embedded_src_segments[1].sum(dim=1))

        #############################################################################################
        # print("embedded_token_positions[0].size()", embedded_token_positions[0].size())
        # print("embedded_token_positions[0].sum(dim=1)", embedded_token_positions[0].sum(dim=1))

        # print("embedded_token_positions[1].size()", embedded_token_positions[1].size())
        # print("embedded_token_positions[1].sum(dim=1)", embedded_token_positions[1].sum(dim=1))

        # word positions
        if not self.no_word_pos:
            embedded_word_positions = self.embed_positions(
                input=src_word_pos, # needed incase seq_len too long and need to increase size of embedding table
                positions=src_word_pos # the actual positions to encode
            )
            morphed_word_positions = self.morph_word_pos(embedded_word_positions)
            x += self.word_pos_emb_alpha * morphed_word_positions

        #############################################################################################
        # print("embedded_word_positions[0].size()", embedded_word_positions[0].size())
        # print("embedded_word_positions[0].sum(dim=1)", embedded_word_positions[0].sum(dim=1))

        # print("embedded_word_positions[1].size()", embedded_word_positions[1].size())
        # print("embedded_word_positions[1].sum(dim=1)", embedded_word_positions[1].sum(dim=1))

        ################################################################################################################
        # Pass inputs through model

        # input dropout
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        #############################################################################################
        graphemes_and_speechreps_padding_mask = src_tokens.eq(self.padding_idx)  # False for grapheme+speech timesteps, True for padding timesteps

        # print("graphemes_and_speechreps_padding_mask[0].size()", graphemes_and_speechreps_padding_mask[0].size())
        # print("graphemes_and_speechreps_padding_mask[0]", graphemes_and_speechreps_padding_mask[0])
        # print("graphemes_and_speechreps_padding_mask[0].sum(dim=0)", graphemes_and_speechreps_padding_mask[0].sum(dim=0))

        # print("graphemes_and_speechreps_padding_mask[1].size()", graphemes_and_speechreps_padding_mask[1].size())
        # print("graphemes_and_speechreps_padding_mask[1]", graphemes_and_speechreps_padding_mask[1])
        # print("graphemes_and_speechreps_padding_mask[1].sum(dim=0)", graphemes_and_speechreps_padding_mask[1].sum(dim=0))

        for layer in self.transformer_layers:
            x = layer(x, graphemes_and_speechreps_padding_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.embed_speaker is not None:
            seq_len, bsz, _ = x.size()
            emb = self.embed_speaker(speaker).transpose(0, 1)
            emb = emb.expand(seq_len, bsz, -1)
            x = self.spk_emb_proj(torch.cat([x, emb], dim=2))

        ################################################################################################################
        # Create a new padding mask for the transformer decoder to limit what timesteps of encoder out it can attend over
        if self.dont_mask_encoder_out_speech_timesteps:
            # decoder can attend over grapheme+speechreps timesteps
            encoder_padding_mask = graphemes_and_speechreps_padding_mask
        else:
            # pads out the speechreps sequence so that
            # the transformer decoder cannot attend over those output timesteps, and instead can only attend
            # over timesteps that correspond to the input graphemes
            grapheme_segment_idx = 1 # pad idx is 0, speechreps idx is 2
            graphemes_padding_mask = src_segments.ne(grapheme_segment_idx) # pad everything BUT the graphemes (1 indicate to pad a position)
            encoder_padding_mask = graphemes_padding_mask

        # print("encoder_padding_mask[1]", encoder_padding_mask[1])
        # print("encoder_padding_mask[1].size()", encoder_padding_mask[1].size())
        # print("encoder_padding_mask[1].sum(dim=0)", encoder_padding_mask[1].sum(dim=0))

        return {
            "encoder_out": [x],  # T x B x C

            # in TransformerDecoderLayerBase.forward()
            # masks out what timesteps decoder can attend to from encoder outs
            #       encoder_padding_mask(ByteTensor, optional): binary
            #       ByteTensor of shape `(batch, src_len)` where padding
            #       elements are indicated by ``1``.
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


def decoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("tanh"))


class TTSTransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, src_dict):
        super().__init__(None)
        self._future_mask = torch.empty(0)

        self.args = args
        self.padding_idx = src_dict.pad()
        self.n_frames_per_step = args.n_frames_per_step
        self.out_dim = args.output_frame_dim * args.n_frames_per_step

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, args.decoder_embed_dim, self.padding_idx
        )
        self.pos_emb_alpha = nn.Parameter(torch.ones(1))
        self.prenet = nn.Sequential(
            Prenet(self.out_dim, args.prenet_layers, args.prenet_dim,
                   args.prenet_dropout),
            nn.Linear(args.prenet_dim, args.decoder_embed_dim),
        )

        self.n_transformer_layers = args.decoder_transformer_layers
        self.transformer_layers = nn.ModuleList(
            TransformerDecoderLayer(args)
            for _ in range(self.n_transformer_layers)
        )
        if args.decoder_normalize_before:
            self.layer_norm = LayerNorm(args.decoder_embed_dim)
        else:
            self.layer_norm = None

        self.feat_proj = nn.Linear(args.decoder_embed_dim, self.out_dim)
        self.eos_proj = nn.Linear(args.decoder_embed_dim, 1)

        self.postnet = Postnet(self.out_dim, args.postnet_conv_dim,
                               args.postnet_conv_kernel_size,
                               args.postnet_layers, args.postnet_dropout)

        self.ctc_proj = None
        if getattr(args, "ctc_weight", 0.) > 0.:
            self.ctc_proj = nn.Linear(self.out_dim, len(src_dict))

        self.apply(decoder_init)

    def extract_features(
            self, prev_outputs, encoder_out=None, incremental_state=None,
            target_lengths=None, speaker=None, **kwargs
    ):
        alignment_layer = self.n_transformer_layers - 1
        self_attn_padding_mask = lengths_to_padding_mask(target_lengths)
        positions = self.embed_positions(
            self_attn_padding_mask, incremental_state=incremental_state
        )

        # print(f"inside sac_transformer.py extract feats() prev_outputs is size()", prev_outputs.size())
        # print(f"inside sac_transformer.py extract feats() incremental_state is", incremental_state)

        if incremental_state is not None:
            prev_outputs = prev_outputs[:, -1:, :]
            self_attn_padding_mask = self_attn_padding_mask[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        x = self.prenet(prev_outputs)
        x += self.pos_emb_alpha * positions
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if not self_attn_padding_mask.any():
            self_attn_padding_mask = None

        attn: Optional[torch.Tensor] = None
        inner_states: List[Optional[torch.Tensor]] = [x]
        for idx, transformer_layer in enumerate(self.transformer_layers):
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = transformer_layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            # average probabilities over heads, transpose to
            # (B, src_len, tgt_len)
            attn = attn.mean(dim=0).transpose(2, 1)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward(self, prev_output_tokens, encoder_out=None,
                incremental_state=None, target_lengths=None, speaker=None,
                **kwargs):
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out,
            incremental_state=incremental_state, target_lengths=target_lengths,
            speaker=speaker, **kwargs
        )
        attn = extra["attn"]
        feat_out = self.feat_proj(x)
        bsz, seq_len, _ = x.size()
        eos_out = self.eos_proj(x)
        post_feat_out = feat_out + self.postnet(feat_out)
        return post_feat_out, eos_out, {"attn": attn, "feature_out": feat_out}

    def get_normalized_probs(self, net_output, log_probs, sample):
        logits = self.ctc_proj(net_output[2]["feature_out"])
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]


@register_model("sac_transformer")
class SACTransformerModel(FairseqEncoderDecoderModel):
    """
    Implementation for https://arxiv.org/pdf/1809.08895.pdf

    Implementation of Speech Audio Corrector Transformer
      Transformer encoder is modified so that it can also take speechreps as inputs
      Word position information is also provided that aligns graphemes and speechreps that belong to the same word
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument("--dropout", type=float)
        parser.add_argument("--output-frame-dim", type=int)
        parser.add_argument("--speaker-embed-dim", type=int)
        # encoder prenet
        parser.add_argument("--encoder-dropout", type=float)
        parser.add_argument("--encoder-conv-layers", type=int)
        parser.add_argument("--encoder-conv-kernel-size", type=int)
        # encoder transformer layers
        parser.add_argument("--encoder-transformer-layers", type=int)
        parser.add_argument("--encoder-embed-dim", type=int)
        parser.add_argument("--encoder-ffn-embed-dim", type=int)
        parser.add_argument("--encoder-normalize-before", action="store_true")
        parser.add_argument("--encoder-attention-heads", type=int)
        parser.add_argument("--attention-dropout", type=float)
        parser.add_argument("--activation-dropout", "--relu-dropout", type=float)
        parser.add_argument("--activation-fn", type=str, default="relu")
        # Encoder outputs masking
        parser.add_argument("--dont-mask-encoder-out-speech-timesteps", action='store_true',
                            help='by default decoder can attend to encoder timesteps of graphemes and speech,'
                                 'can optionally only attend to grapheme timesteps by masking speech timesteps.')
        parser.add_argument("--no-word-pos", action='store_true',
                            help='do not use word positional information, likely to hinder test time correction perf')
        # decoder prenet
        parser.add_argument("--prenet-dropout", type=float)
        parser.add_argument("--prenet-layers", type=int)
        parser.add_argument("--prenet-dim", type=int)
        # decoder postnet
        parser.add_argument("--postnet-dropout", type=float)
        parser.add_argument("--postnet-layers", type=int)
        parser.add_argument("--postnet-conv-dim", type=int)
        parser.add_argument("--postnet-conv-kernel-size", type=int)
        # decoder transformer layers
        parser.add_argument("--decoder-transformer-layers", type=int)
        parser.add_argument("--decoder-embed-dim", type=int)
        parser.add_argument("--decoder-ffn-embed-dim", type=int)
        parser.add_argument("--decoder-normalize-before", action="store_true")
        parser.add_argument("--decoder-attention-heads", type=int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_updates = 0

    @classmethod
    def build_model(cls, args, task):
        embed_speaker = task.get_speaker_embeddings(args)
        encoder = SACTransformerEncoder(args, task.src_dict, embed_speaker)
        decoder = TTSTransformerDecoder(args, task.src_dict)
        return cls(encoder, decoder)

    def forward(self, src_tokens, src_token_pos, src_word_pos, src_segments, src_lengths, prev_output_tokens, **kwargs):
        """
        SAC (Speech Audio Corrector):
            override forward in FairseqEncoderDecoderModel in order to provide additional inputs for SAC
            i.e. src_word_pos, src_segments
        """
        encoder_out = self.encoder(src_tokens, src_token_pos=src_token_pos, src_word_pos=src_word_pos,
                                   src_segments=src_segments, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out

    def forward_encoder(self, src_tokens, src_token_pos, src_word_pos, src_segments, src_lengths, speaker=None, **kwargs):
        return self.encoder(src_tokens, src_token_pos=src_token_pos, src_word_pos=src_word_pos,
                            src_segments=src_segments, src_lengths=src_lengths, speaker=speaker, **kwargs)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self._num_updates = num_updates


@register_model_architecture("sac_transformer", "sac_transformer")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.output_frame_dim = getattr(args, "output_frame_dim", 80)
    args.speaker_embed_dim = getattr(args, "speaker_embed_dim", 64)
    # encoder prenet
    args.encoder_dropout = getattr(args, "encoder_dropout", 0.5)
    args.encoder_conv_layers = getattr(args, "encoder_conv_layers", 3)
    args.encoder_conv_kernel_size = getattr(args, "encoder_conv_kernel_size", 5)
    # encoder transformer layers
    args.encoder_transformer_layers = getattr(args, "encoder_transformer_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * args.encoder_embed_dim)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    # Encoder outputs masking
    args.dont_mask_encoder_out_speech_timesteps = getattr(args, "dont_mask_encoder_out_speech_timesteps", False)
    args.no_word_pos = getattr(args, "no_word_pos", False)
    # decoder prenet
    args.prenet_dropout = getattr(args, "prenet_dropout", 0.5)
    args.prenet_layers = getattr(args, "prenet_layers", 2)
    args.prenet_dim = getattr(args, "prenet_dim", 256)
    # decoder postnet
    args.postnet_dropout = getattr(args, "postnet_dropout", 0.5)
    args.postnet_layers = getattr(args, "postnet_layers", 5)
    args.postnet_conv_dim = getattr(args, "postnet_conv_dim", 512)
    args.postnet_conv_kernel_size = getattr(args, "postnet_conv_kernel_size", 5)
    # decoder transformer layers
    args.decoder_transformer_layers = getattr(args, "decoder_transformer_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4 * args.decoder_embed_dim)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
