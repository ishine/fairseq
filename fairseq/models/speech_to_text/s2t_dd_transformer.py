#!/usr/bin/env python3

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils
from fairseq.models import (
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding
from fairseq.models.transformer_dd import TransformerDualDecoder
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerModel,
    TransformerDecoderScriptable,
)
from torch import Tensor


logger = logging.getLogger(__name__)


class MultiOutputDecoder(FairseqDecoder):
    def __init__(
        self, 
        dictionary,
        decoder_asr,
        decoder_st,
    ):
        super().__init__(dictionary)
        self.decoder_asr = decoder_asr
        self.decoder_st = decoder_st

    def forward(
        self, 
        prev_output_tokens, 
        encoder_out,
        **kwargs
    ):
        decoder_asr_out = self.decoder_asr(prev_output_tokens[0], encoder_out)
        decoder_st_out = self.decoder_st(prev_output_tokens[1], encoder_out)
        # logging.info(f'decoder_asr_out[0]: {decoder_asr_out[0]}')
        x = (decoder_asr_out[0], decoder_st_out[0])
        attn = (
            decoder_asr_out[1]["attn"], decoder_st_out[1]["attn"]
        )
        inner_states = (
            decoder_asr_out[1]["inner_states"], decoder_st_out[1]["inner_states"]
        )
        return x, {"attn": [attn], "inner_states": inner_states}


@register_model("s2t_dd_transformer")
class S2TDualDecoderTransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # input
        S2TTransformerModel.add_args(parser)
        
        # args for Dual-decoder Transformer model from `"Dual-decoder Transformer for 
        # Joint ASR and Multilingual ST (Le et al., 2020)
        parser.add_argument('--merge-operator', type=str, metavar='STR', default=None,
                            choices=["sum", "concat"],
                            help="Operator used when merging dual-attention with main branch")
        parser.add_argument('--dual-attn-position', type=str, metavar='STR', default=None,
                            choices=["at-self", "at-source", "at-self-and-source"],
                            help="Position of the dual-attention module")
        parser.add_argument('--merge-sum-weight-init', type=float, metavar='D', default=0.0,
                            help="Init weight for sum merging operator")
        parser.add_argument('--wait-k', default="0", type=str, metavar='STR',
                            help="[start:stop:step] for k, which is the number of steps \
                                that ASR decoder is ahead of ST decoder.")
        parser.add_argument('--dual-attn-lang', action="store_true",
                            help="Use language-specific dual-attn layers.")
        parser.add_argument('--dual-lang-pairs', type=str, default=None,
                            help="Language pairs in training, separated by comma.\
                            Required if --dual-attn-lang is set to True")
        parser.add_argument('--load-pretrain-speech-encoder',
                            type=str,
                            default="",
                            metavar="EXPR",
                            help=""" path to the pretrained speech encoder """,
                        )
        parser.add_argument('--single-decoder', action="store_true",
                            help="Use single decoder for the two outputs")

    @classmethod
    def build_encoder(cls, args):
        encoder = S2TTransformerModel.build_encoder(args)
        if getattr(args, "load_pretrain_speech_encoder", "") != "":
            logging.info(f"Loading pretrained speech encoder ...")
            state = checkpoint_utils.load_checkpoint_to_cpu(args.load_pretrain_speech_encoder)
            ckpt_name = "encoder.spch_encoder"
            ckpt_component_type = [ckpt_name] if any([key.startswith(ckpt_name) \
                                    for key in state["model"].keys()]) else ["encoder"]
            checkpoint_utils.load_pretrained_component_from_model_different_keys(
                        encoder, state, ckpt_component_types=ckpt_component_type
                        )
            logging.info(f"Loaded pretrained speech encoder from {args.load_pretrain_speech_encoder}")
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        if getattr(args, "single_decoder", False):
            decoder_asr = TransformerDecoderScriptable(
                args, task.target_dictionary, embed_tokens
            )
            decoder_st = TransformerDecoderScriptable(
                args, task.target_dictionary, embed_tokens
            )
            decoder_asr = decoder_st
            decoder = MultiOutputDecoder(task.target_dictionary, decoder_asr, decoder_st)
            return decoder
            
        decoder = TransformerDualDecoderScriptable(args, task.target_dictionary, embed_tokens)
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        # get names of subtasks
        args.subtasks = getattr(args, "subtasks", None)
        if not isinstance(args.subtasks, list):
            args.subtasks = args.subtasks.split("_") if args.subtasks is not None else None

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        if getattr(args, "single_decoder", False):
            decoder_embed_tokens = build_embedding(
                task.target_dictionary, args.decoder_embed_dim
            )
        else:
            decoder_embed_tokens = nn.ModuleDict({k: build_embedding(
                task.target_dictionary, args.decoder_embed_dim
            ) for k in args.subtasks})
            
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        for i in range(2):
            lprobs[i].batch_first = True
        return lprobs

    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        logits = tuple([net_output[0][i].float() for i in range(2)])
        if log_probs:
            return tuple([F.log_softmax(logits[i], dim=-1) for i in range(2)])
        else:
            return tuple([F.softmax(logits, dim=-1) for i in range(2)])

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return tuple([sample["target"][i] for i in range(2)])

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class TransformerDualDecoderScriptable(TransformerDualDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Tuple[Dict[str, Dict[str, Optional[Tensor]]]]] = (None, None),
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, None


@register_model_architecture(model_name="s2t_dd_transformer", arch_name="s2t_dd_transformer_base")
def base_architecture(args):
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)


@register_model_architecture("s2t_dd_transformer", "s2t_dd_transformer_xxs")
def s2t_dd_transformer_xxs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("s2t_dd_transformer", "s2t_dd_transformer_xs")
def s2t_dd_transformer_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("s2t_dd_transformer", "s2t_dd_transformer_s")
def s2t_dd_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_dd_transformer", "s2t_dd_transformer_sp")
def s2t_dd_transformer_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_dd_transformer_s(args)


@register_model_architecture("s2t_dd_transformer", "s2t_dd_transformer_m")
def s2t_dd_transformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_dd_transformer", "s2t_dd_transformer_mp")
def s2t_dd_transformer_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_dd_transformer_m(args)


@register_model_architecture("s2t_dd_transformer", "s2t_dd_transformer_l")
def s2t_dd_transformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("s2t_dd_transformer", "s2t_dd_transformer_lp")
def s2t_dd_transformer_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_dd_transformer_l(args)
