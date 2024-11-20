# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor


class TransformerDualDecoderLayer(nn.Module):
    """Dual-Decoder layer block.

    Dual-decoder Transformer layer from `"Dual-decoder Transformer for Joint 
    Automatic Speech Recognition and Multilingual Speech Translation" 
    (Le et al, 2020) <https://www.aclweb.org/anthology/2020.coling-main.314/>`_.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim

        # arguments for "Dual-decoder Transformer for Joint ASR and Multilingual ST"
        self.merge_operator = getattr(args, "merge_operator", None)
        self.dual_attn_position = getattr(args, "dual_attn_position", None)
        self.subtasks = getattr(args, "subtasks", None)
        merge_sum_weight_init = getattr(args, "merge_sum_weight_init", 0.0)
        self.dual_attn_lang = getattr(args, "dual_attn_lang", False)
        dual_lang_pairs = getattr(args, "dual_lang_pairs", None)
        self.shared_dual_attn = True if not self.dual_attn_lang else False
        if dual_lang_pairs is not None:
            dual_lang_pairs = dual_lang_pairs.split(",")
            src_keys = [f'{self.subtasks[0]}_<lang:{p.split("-")[0]}>' for p in dual_lang_pairs]
            tgt_keys = [f'{self.subtasks[1]}_<lang:{p.split("-")[1]}>' for p in dual_lang_pairs]
            self.dual_attn_names = src_keys + tgt_keys
        else: # for backward compatibility
            self.dual_attn_names = self.subtasks
            self.shared_dual_attn = False
        logging.info(f'self.dual_attn_names: {self.dual_attn_names}')
        logging.info(f'self.shared_dual_attn: {self.shared_dual_attn}')

        self.dropout_module = nn.ModuleDict({k: FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        ) for k in self.subtasks})
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = nn.ModuleDict(
            {k: self.build_self_attention(
                    self.embed_dim,
                    args,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn) for k in self.subtasks})

        self.dual_attn_at_self, self.dual_attn_at_src = None, None
        if self.dual_attn_position is not None:
            if "self" in self.dual_attn_position:
                # self.dual_attn_at_self = nn.ModuleDict({k: self.build_dual_attention(
                #     self.embed_dim, args) for k in self.subtasks})
                self.dual_attn_at_self = self.build_dual_attention(
                    self.dual_attn_names,
                    self.embed_dim,
                    args,
                    shared=self.shared_dual_attn,
                )
            if "source" in self.dual_attn_position:
                # self.dual_attn_at_src = nn.ModuleDict({k: self.build_dual_attention(
                #     self.embed_dim, args) for k in self.subtasks})
                self.dual_attn_at_src = self.build_dual_attention(
                    self.dual_attn_names,
                    self.embed_dim,
                    args,
                    shared=self.shared_dual_attn,
                )

        if self.merge_operator is not None:
            if self.merge_operator == "sum":
                if "self" in self.dual_attn_position:
                    self.merge_sum_weight_self = nn.ParameterDict({k: torch.nn.Parameter(
                        torch.tensor(merge_sum_weight_init)) for k in self.dual_attn_names})
                if "source" in self.dual_attn_position:
                    self.merge_sum_weight_source = nn.ParameterDict({k: torch.nn.Parameter(
                    torch.tensor(merge_sum_weight_init)) for k in self.dual_attn_names})
            elif self.merge_operator == "concat":
                if "self" in self.dual_attn_position:
                    self.merge_concat_self = nn.ModuleDict({k: nn.Linear(
                        self.embed_dim*2, self.embed_dim) for k in self.dual_attn_names})
                if "source" in self.dual_attn_position:
                    self.merge_concat_source = nn.ModuleDict({k: nn.Linear(
                        self.embed_dim*2, self.embed_dim) for k in self.dual_attn_names})

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = nn.ModuleDict({k: FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        ) for k in self.subtasks})
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = nn.ModuleDict({k: LayerNorm(
            self.embed_dim, export=export) for k in self.subtasks})

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = nn.ModuleDict({
                k: self.build_encoder_attention(self.embed_dim, args)
                for k in self.subtasks})
            self.encoder_attn_layer_norm = nn.ModuleDict({
                k: LayerNorm(self.embed_dim, export=export)
                for k in self.subtasks})

        self.fc1 = nn.ModuleDict({k: self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        ) for k in self.subtasks})
        self.fc2 = nn.ModuleDict({k: self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        ) for k in self.subtasks})

        self.final_layer_norm = nn.ModuleDict({k: LayerNorm(
            self.embed_dim, export=export) for k in self.subtasks})
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
    def _build_dual_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_dual_attention(self, dual_attn_names, embed_dim, args, shared=False):
        if not dual_attn_names:
            return None
        
        if shared:
            dual_attn = self._build_dual_attention(embed_dim, args)
            return nn.ModuleDict({k: dual_attn for k in dual_attn_names})

        return nn.ModuleDict({k: self._build_dual_attention(embed_dim, args)
                                    for k in dual_attn_names})

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Tuple[Dict[str, Dict[str, Optional[Tensor]]]]] = (None, None),
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[Tuple[torch.Tensor]] = (None, None),
        self_attn_padding_mask: Optional[Tuple[torch.Tensor]] = (None, None),
        dual_attn_mask: Optional[Tuple[torch.Tensor]] = (None, None),
        dual_attn_padding_mask: Optional[Tuple[torch.Tensor]] = (None, None),
        need_attn: bool = False,
        need_head_weights: bool = False,
        dual_attn_names: Optional[List[str]] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        ntask = len(self.subtasks)
        if need_head_weights:
            need_attn = True

        residual = x
        y = x
        if self.normalize_before:
            x = tuple([self.self_attn_layer_norm[task](x[i]) for i, task in enumerate(self.subtasks)])
            y = x
        if prev_self_attn_state is not None: # TODO: to revise for dual-decoder Transformer
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert None not in incremental_state
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = tuple([self.self_attn[task]._get_input_buffer(incremental_state[i]) 
                                    for i, task in enumerate(self.subtasks)])

        if self.cross_self_attention and not (
            None not in incremental_state
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask != (None, None):
                assert encoder_out is not None
                self_attn_mask = tuple([torch.cat(
                    (x[i].new_zeros(x[i].size(0), encoder_out.size(0)), self_attn_mask[i]), dim=1
                ) for i in range(ntask)])
            if self_attn_padding_mask != (None, None):
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = tuple([self_attn_padding_mask[i].new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    ) for i in range(ntask)])
                self_attn_padding_mask = tuple([torch.cat(
                    (encoder_padding_mask[i], self_attn_padding_mask[i]), dim=1
                ) for i in range(ntask)])
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x_tmp, attn_tmp = [None] * ntask, [None] * ntask
        for i, task in enumerate(self.subtasks):
            x_tmp[i], attn_tmp[i] = self.self_attn[task](
                query=x[i],
                key=y[i],
                value=y[i],
                key_padding_mask=self_attn_padding_mask[i],
                incremental_state=incremental_state[i],
                need_weights=False,
                attn_mask=self_attn_mask[i],
            )
        x, attn = tuple(x_tmp), tuple(attn_tmp)
    
        # Dual-attention layer at self
        if self.dual_attn_at_self is not None:
            x_tmp, attn_tmp = [None] * ntask, [None] * ntask
            for i, task in enumerate(dual_attn_names):
                z, _ = self.dual_attn_at_self[task](
                    query=y[i],
                    key=y[1-i],
                    value=y[1-i],
                    key_padding_mask=dual_attn_padding_mask[i],
                    incremental_state=incremental_state[i],
                    need_weights=False,
                    attn_mask=dual_attn_mask[i],
                    wait_k=True,
                )
                if self.merge_operator == "sum":
                    x_tmp[i] = x[i] + self.merge_sum_weight_self[task] * z
                elif self.merge_operator == "concat":
                    x_tmp[i] = self.merge_concat_self[task](torch.cat((x[i], z), dim=-1))
                else:
                    raise NotImplementedError
            x = tuple(x_tmp)
    
        x = tuple([self.dropout_module[task](x[i]) for i, task in enumerate(self.subtasks)])
        x = tuple([self.residual_connection(x[i], residual[i]) for i, task in enumerate(self.subtasks)])
        if not self.normalize_before:
            x = tuple([self.self_attn_layer_norm[task](x[i]) for i, task in enumerate(self.subtasks)])

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            y = x
            if self.normalize_before:
                x = tuple([self.encoder_attn_layer_norm[task](x[i]) for i, task in enumerate(self.subtasks)])
                y = x
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x_tmp, attn_tmp = [None] * ntask, [None] * ntask
            for i, task in enumerate(self.subtasks):
                x_tmp[i], attn_tmp[i] = self.encoder_attn[task](
                    query=x[i],
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state[i],
                    static_kv=True,
                    need_weights=need_attn or (not self.training and self.need_attn),
                    need_head_weights=need_head_weights,
                )
            x , attn = tuple(x_tmp), tuple(attn_tmp)

            # Dual-attention layer at encoder-decoder attention (called source attention)
            if self.dual_attn_at_src is not None:
                x_tmp, attn_tmp = [None] * ntask, [None] * ntask
                for i, task in enumerate(dual_attn_names):
                    z, _ = self.dual_attn_at_src[task](
                        query=y[i],
                        key=y[1-i],
                        value=y[1-i],
                        key_padding_mask=dual_attn_padding_mask[i],
                        incremental_state=incremental_state[i],
                        need_weights=False,
                        attn_mask=dual_attn_mask[i],
                        wait_k=True,
                    )
                    if self.merge_operator == "sum":
                        x_tmp[i] = x[i] + self.merge_sum_weight_source[task] * z
                    elif self.merge_operator == "concat":
                        x_tmp[i] = self.merge_concat_source[task](torch.cat((x[i], z), dim=-1))
                    else:
                        raise NotImplementedError
                x = tuple(x_tmp)

            x = tuple([self.dropout_module[task](x[i]) for i, task in enumerate(self.subtasks)])
            x = tuple([self.residual_connection(x[i], residual[i]) for i, task in enumerate(self.subtasks)])
            if not self.normalize_before:
                x = tuple([self.encoder_attn_layer_norm[task](x[i]) for i, task in enumerate(self.subtasks)])

        residual = x
        if self.normalize_before:
            x = tuple([self.final_layer_norm[task](x[i]) for i, task in enumerate(self.subtasks)])

        x = tuple([self.activation_fn(self.fc1[task](x[i])) for i, task in enumerate(self.subtasks)])
        x = tuple([self.activation_dropout_module[task](x[i]) for i, task in enumerate(self.subtasks)])
        x = tuple([self.fc2[task](x[i]) for i, task in enumerate(self.subtasks)])
        x = tuple([self.dropout_module[task](x[i]) for i, task in enumerate(self.subtasks)])
        x = tuple([self.residual_connection(x[i], residual[i]) for i, task in enumerate(self.subtasks)])
        if not self.normalize_before:
            x = tuple([self.final_layer_norm[task](x[i]) for i, task in enumerate(self.subtasks)])
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]

            return x, attn, self_attn_state

        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn