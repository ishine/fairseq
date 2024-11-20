# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from typing import Any, Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.transformer import TransformerEncoder, TransformerModel, base_architecture
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDualDecoderLayer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset
from torch import Tensor


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("dd_transformer")
class DualDecoderTransformerModel(FairseqEncoderDecoderModel):
    """
    Dual-decoder Transformer model from `"Dual-decoder Transformer for Joint 
    Automatic Speech Recognition and Multilingual Speech Translation" 
    (Le et al, 2020) <https://www.aclweb.org/anthology/2020.coling-main.314/>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDualDecoder): the dual-decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        TransformerModel.add_args(parser)
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDualDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class TransformerDualDecoder(FairseqIncrementalDecoder):
    """
    Dual-decoder Transformer consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDualDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = (torch.empty(0), torch.empty(0))
        self._future_mask_dual = (torch.empty(0), torch.empty(0))
        self.subtasks = getattr(args, "subtasks", None)
        self.merge_operator = getattr(args, "merge_operator", None)
        wait_k = getattr(args, "wait_k", "0")
        wait_k = wait_k.replace("[", "").replace("]", "").split(":")
        assert len(wait_k) <= 3
        if len(wait_k) == 1:
            self.wait_k = int(wait_k[0])
        else:
            start, stop = int(wait_k[0]), int(wait_k[1])
            step = 1 if len(wait_k) == 2 else int(wait_k[-1])
            wait_k = list(range(start, stop, step))
            self.wait_k = wait_k
        logging.info(f'self.wait_k: {self.wait_k}')

        self.dual_lang_pairs = getattr(args, "dual_lang_pairs", None)
        self.lang_token_ids = {
            i: s
            for s, i in dictionary.indices.items() if SpeechToTextDataset.is_lang_tag(s)
        }
        logging.info(f'lang_token_ids: {self.lang_token_ids}')

        self.dropout_module = nn.ModuleDict({k: FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        ) for k in self.subtasks})
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens[self.subtasks[0]].embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens[self.subtasks[0]].padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = nn.ModuleDict(
            {k: Linear(input_embed_dim, embed_dim, bias=False) for k in self.subtasks}
            ) if embed_dim != input_embed_dim else None

        self.embed_positions = nn.ModuleDict(
            {k: PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            ) for k in self.subtasks}) if not args.no_token_positional_embeddings else None

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = nn.ModuleDict({k: LayerNorm(embed_dim) for k in self.subtasks})
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_dual_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = nn.ModuleDict({k: LayerNorm(embed_dim) for k in self.subtasks})
        else:
            self.layer_norm = None

        self.project_out_dim = nn.ModuleDict(
            {k: Linear(embed_dim, self.output_embed_dim, bias=False) for k in self.subtasks}
        ) if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.ModuleDict({k: nn.Linear(
                self.embed_tokens[k].weight.shape[1],
                self.embed_tokens[k].weight.shape[0],
                bias=False,
            ) for k in self.subtasks})
            for k in self.subtasks:
                self.output_projection[k].weight = self.embed_tokens[k].weight
        else:
            self.output_projection = nn.ModuleDict({k: nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            ) for k in self.subtasks})
            for k in self.subtasks:
                nn.init.normal_(
                    self.output_projection[k].weight, mean=0, std=self.output_embed_dim ** -0.5
                )

    def build_dual_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDualDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Tuple[Dict[str, Dict[str, Optional[Tensor]]]]] = (None, None),
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Tuple[Dict[str, Dict[str, Optional[Tensor]]]]] = (None, None),
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Tuple[Dict[str, Dict[str, Optional[Tensor]]]]] = (None, None),
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        ntask = len(self.subtasks)
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1
        
        dual_attn_names = self.subtasks
        if self.dual_lang_pairs is not None:
            # if prev_output_tokens[0][:, 1:2].shape[1] != 0:
            src_lang_id = prev_output_tokens[0][:, 1:2][0].item()
            tgt_lang_id = prev_output_tokens[1][:, 1:2][0].item()
            src_lang_key = f'{self.subtasks[0]}_{self.lang_token_ids[src_lang_id]}'
            tgt_lang_key = f'{self.subtasks[1]}_{self.lang_token_ids[tgt_lang_id]}'
            dual_attn_names = [src_lang_key, tgt_lang_key]

        # embed positions
        positions = tuple(
            [self.embed_positions[task](
                prev_output_tokens[i], incremental_state=incremental_state[i]) 
                for i, task in enumerate(self.subtasks)]) if self.embed_positions is not None else None

        if incremental_state != (None, None):
            prev_output_tokens = tuple([prev_output_tokens[i][:, -1:] for i in range(ntask)])
            if positions is not None:
                positions = tuple([positions[i][:, -1:] for i in range(ntask)])

        # embed tokens and positions
        x = tuple([self.embed_scale * self.embed_tokens[task](prev_output_tokens[i]) for i, task in enumerate(self.subtasks)])

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = tuple([self.project_in_dim[task](x[i]) for i, task in enumerate(self.subtasks)])

        if positions is not None:
            x_tmp = [None] * ntask
            for i in range(ntask):
                x_tmp[i] = x[i] + positions[i]
            x = tuple(x_tmp)

        if self.layernorm_embedding is not None:
            x = tuple([self.layernorm_embedding[task](x[i]) for i, task in enumerate(self.subtasks)])

        x = tuple([self.dropout_module[task](x[i]) for i, task in enumerate(self.subtasks)])

        # B x T x C -> T x B x C
        x_tmp = [None] * ntask
        for i in range(ntask):
            x_tmp[i] = x[i].transpose(0, 1)
        x = tuple(x_tmp)

        self_attn_padding_mask: Optional[List[Tensor]] = [None, None]
        dual_attn_padding_mask: Optional[List[Tensor]] = [None, None]
        if self.cross_self_attention or \
            any([prev_output_tokens[i].eq(self.padding_idx).any() \
                for i in range(ntask)]):
            for i in range(ntask):
                self_attn_padding_mask[i] = prev_output_tokens[i].eq(self.padding_idx)
                dual_attn_padding_mask[i] = prev_output_tokens[1-i].eq(self.padding_idx)

        # dual-decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state == (None, None) and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
                dual_attn_mask = self.buffered_future_mask(x, dual_attn=True)
            else:
                self_attn_mask = (None, None)
                dual_attn_mask = (None, None)

            x, layer_attn, _ = layer(
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
                dual_attn_mask=dual_attn_mask,
                dual_attn_padding_mask=dual_attn_padding_mask,
                dual_attn_names=dual_attn_names,
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn_tmp = [None] * ntask
                for i in range(ntask):
                    attn_tmp[i] = layer_attn[i].float().to(x[i])
                attn = tuple(attn_tmp)

        if attn is not None:
            if alignment_heads is not None:
                attn_tmp = [None] * ntask
                for i in range(ntask):
                    attn_tmp[i] = attn[i][:alignment_heads]
                attn = tuple(attn_tmp)

            # average probabilities over heads
            attn_tmp = [None] * ntask
            for i in range(ntask):
                attn_tmp[i] = attn[i].mean(dim=0)
            attn = tuple(attn_tmp)

        if self.layer_norm is not None:
            x = tuple([self.layer_norm[task](x[i]) for i, task in enumerate(self.subtasks)])

        # T x B x C -> B x T x C
        x_tmp = [None] * ntask
        for i in range(ntask):
            x_tmp[i] = x[i].transpose(0, 1)
        x = tuple(x_tmp)

        if self.project_out_dim is not None:
            x = tuple([self.project_out_dim[task](x[i]) for i, task in enumerate(self.subtasks)])

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return tuple([self.output_projection[task](features[i]) for i, task in enumerate(self.subtasks)])
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min([self.max_target_positions] + 
            [self.embed_positions[task].max_positions for i, task in enumerate(self.subtasks)])

    def buffered_future_mask(self, tuple_tensor, dual_attn=False):
        ntask = len(self.subtasks)
        dim = tuple([tuple_tensor[i].size(0) for i in range(ntask)])
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        _future_mask_tmp = [self._future_mask[i] for i in range(ntask)]
        if not dual_attn:
            for i in range(ntask):
                if (
                    self._future_mask[i].size(0) == 0
                    or (not self._future_mask[i].device == tuple_tensor[i].device)
                    or self._future_mask[i].size(0) < dim[i]
                ):
                    _future_mask_tmp[i] = torch.triu(
                        utils.fill_with_neg_inf(torch.zeros([dim[i], dim[i]])), 1
                    )
                _future_mask_tmp[i] = _future_mask_tmp[i].to(tuple_tensor[i])
            self._future_mask = tuple(_future_mask_tmp)
            return tuple([self._future_mask[i][:dim[i], :dim[i]] for i in range(ntask)])
        else:
            wait_k = random.choice(self.wait_k) if isinstance(self.wait_k, list) else self.wait_k
            for i in range(ntask):
                diagonal = 1 if wait_k==0 else -wait_k if i == 0 else wait_k + 1
                _future_mask_tmp[i] = torch.triu(
                        utils.fill_with_neg_inf(torch.zeros([dim[i], dim[1-i]])),
                        diagonal=diagonal
                    )
                _future_mask_tmp[i] = _future_mask_tmp[i].to(tuple_tensor[i])

            self._future_mask_dual = tuple(_future_mask_tmp)
            return tuple([self._future_mask_dual[i][:dim[i], :dim[1-i]] for i in range(ntask)])

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m