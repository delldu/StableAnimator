# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect

from collections import OrderedDict
from functools import partial
from typing import Dict, Optional, Tuple, Union, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from diffusers.models.modeling_utils import ModelMixin

from dataclasses import dataclass, fields, is_dataclass

from diffusers.utils.torch_utils import randn_tensor

from diffusers.utils import deprecate, is_torch_version, logging

from diffusers.models.activations import get_activation
from diffusers.utils.import_utils import is_peft_available, is_torch_available, is_transformers_available


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


import pdb
import todos

# ------------------------------
class Attention(nn.Module):
    r"""
        in_channels,
        heads=in_channels // attention_head_dim,
        dim_head=attention_head_dim,
        rescale_output_factor=output_scale_factor,
        eps=resnet_eps,
        norm_num_groups=attn_groups,
        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
        residual_connection=True,
        bias=True,
        upcast_softmax=True,
        -----------------------------------------------------------
        query_dim=in_channels,
        heads=in_channels // attention_head_dim,
        dim_head=attention_head_dim,
        eps=1e-6,
        upcast_attention=upcast_attention,
        norm_num_groups=32,
        bias=True,
        residual_connection=True,
    """
    def __init__(self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        eps: float = 1e-6,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = True,

        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,

        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        cross_attention_dim: Optional[int] = None,
        kv_heads: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        # To prevent circular import.
        # from .normalization import FP32LayerNorm, RMSNorm
        assert cross_attention_norm == None
        assert cross_attention_norm_num_groups == 32
        assert qk_norm == None
        assert added_kv_proj_dim == None
        assert added_proj_bias == True
        assert out_bias == True
        assert scale_qk == True
        assert only_cross_attention == False
        assert processor == None
        assert out_dim == None
        assert context_pre_only == None
        assert pre_only == False
        assert cross_attention_dim == None
        assert kv_heads == None
        assert dropout == 0.0

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        # self.query_dim = query_dim
        # self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection # !!!
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        # self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        # if self.added_kv_proj_dim is None and self.only_cross_attention:
        #     raise ValueError(
        #         "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
        #     )

        if norm_num_groups is not None: # True | False
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        # assert spatial_norm_dim == None
        # if spatial_norm_dim is not None:
        #     self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        # else:
        #     self.spatial_norm = None
        self.spatial_norm = None # !!!

        # assert qk_norm == None
        # if qk_norm is None:
        #     self.norm_q = None
        #     self.norm_k = None
        # elif qk_norm == "layer_norm":
        #     self.norm_q = nn.LayerNorm(dim_head, eps=eps)
        #     self.norm_k = nn.LayerNorm(dim_head, eps=eps)
        # elif qk_norm == "fp32_layer_norm":
        #     self.norm_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
        #     self.norm_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
        # elif qk_norm == "layer_norm_across_heads":
        #     # Lumina applys qk norm across all heads
        #     self.norm_q = nn.LayerNorm(dim_head * heads, eps=eps)
        #     self.norm_k = nn.LayerNorm(dim_head * kv_heads, eps=eps)
        # elif qk_norm == "rms_norm":
        #     self.norm_q = RMSNorm(dim_head, eps=eps)
        #     self.norm_k = RMSNorm(dim_head, eps=eps)
        # else:
        #     raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm'")
        self.norm_q = None
        self.norm_k = None
        # assert cross_attention_norm == None
        # if cross_attention_norm is None:
        #     self.norm_cross = None
        # elif cross_attention_norm == "layer_norm":
        #     self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        # elif cross_attention_norm == "group_norm":
        #     if self.added_kv_proj_dim is not None:
        #         # The given `encoder_hidden_states` are initially of shape
        #         # (batch_size, seq_len, added_kv_proj_dim) before being projected
        #         # to (batch_size, seq_len, cross_attention_dim). The norm is applied
        #         # before the projection, so we need to use `added_kv_proj_dim` as
        #         # the number of channels for the group norm.
        #         norm_cross_num_channels = added_kv_proj_dim
        #     else:
        #         norm_cross_num_channels = self.cross_attention_dim

        #     self.norm_cross = nn.GroupNorm(
        #         num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
        #     )
        # else:
        #     raise ValueError(
        #         f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
        #     )
        self.norm_cross = None


        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        # if self.added_kv_proj_dim is not None:
        #     pdb.set_trace()
        #     self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
        #     self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
        #     if self.context_pre_only is not None:
        #         self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)

        if not self.pre_only: # True ?
            self.to_out = nn.ModuleList([])
            self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))
        else:
            pdb.set_trace()

        # if self.context_pre_only is not None and not self.context_pre_only:
        #     pdb.set_trace()
        #     self.to_add_out = nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)

        # if qk_norm is not None and added_kv_proj_dim is not None:
        #     pdb.set_trace()
        #     if qk_norm == "fp32_layer_norm":
        #         self.norm_added_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
        #         self.norm_added_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
        #     elif qk_norm == "rms_norm":
        #         self.norm_added_q = RMSNorm(dim_head, eps=eps)
        #         self.norm_added_k = RMSNorm(dim_head, eps=eps)
        # else:
        #     self.norm_added_q = None
        #     self.norm_added_k = None

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None:
            processor = AttnProcessor2_0()
        self.set_processor(processor)

    # def set_use_npu_flash_attention(self, use_npu_flash_attention: bool) -> None:
    #     r"""
    #     Set whether to use npu flash attention from `torch_npu` or not.

    #     """
    #     if use_npu_flash_attention:
    #         processor = AttnProcessorNPU()
    #     else:
    #         # set attention processor
    #         # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
    #         # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
    #         # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
    #         processor = (
    #             AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
    #         )
    #     pdb.set_trace()
    #     self.set_processor(processor)

    # def set_use_memory_efficient_attention_xformers(
    #     self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    # ) -> None:
    #     r"""
    #     Set whether to use memory efficient attention from `xformers` or not.

    #     Args:
    #         use_memory_efficient_attention_xformers (`bool`):
    #             Whether to use memory efficient attention from `xformers` or not.
    #         attention_op (`Callable`, *optional*):
    #             The attention operation to use. Defaults to `None` which uses the default attention operation from
    #             `xformers`.
    #     """
    #     pdb.set_trace()

    #     is_custom_diffusion = hasattr(self, "processor") and isinstance(
    #         self.processor,
    #         (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor, CustomDiffusionAttnProcessor2_0),
    #     )
    #     is_added_kv_processor = hasattr(self, "processor") and isinstance(
    #         self.processor,
    #         (
    #             AttnAddedKVProcessor,
    #             AttnAddedKVProcessor2_0,
    #             SlicedAttnAddedKVProcessor,
    #             XFormersAttnAddedKVProcessor,
    #         ),
    #     )

    #     if use_memory_efficient_attention_xformers:
    #         pdb.set_trace()

    #         if is_added_kv_processor and is_custom_diffusion:
    #             raise NotImplementedError(
    #                 f"Memory efficient attention is currently not supported for custom diffusion for attention processor type {self.processor}"
    #             )
    #         if not is_xformers_available():
    #             raise ModuleNotFoundError(
    #                 (
    #                     "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
    #                     " xformers"
    #                 ),
    #                 name="xformers",
    #             )
    #         elif not torch.cuda.is_available():
    #             raise ValueError(
    #                 "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
    #                 " only available for GPU "
    #             )
    #         else:
    #             try:
    #                 # Make sure we can run the memory efficient attention
    #                 _ = xformers.ops.memory_efficient_attention(
    #                     torch.randn((1, 2, 40), device="cuda"),
    #                     torch.randn((1, 2, 40), device="cuda"),
    #                     torch.randn((1, 2, 40), device="cuda"),
    #                 )
    #             except Exception as e:
    #                 raise e

    #         if is_custom_diffusion:
    #             processor = CustomDiffusionXFormersAttnProcessor(
    #                 train_kv=self.processor.train_kv,
    #                 train_q_out=self.processor.train_q_out,
    #                 hidden_size=self.processor.hidden_size,
    #                 cross_attention_dim=self.processor.cross_attention_dim,
    #                 attention_op=attention_op,
    #             )
    #             processor.load_state_dict(self.processor.state_dict())
    #             if hasattr(self.processor, "to_k_custom_diffusion"):
    #                 processor.to(self.processor.to_k_custom_diffusion.weight.device)
    #         elif is_added_kv_processor:
    #             # TODO(Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
    #             # which uses this type of cross attention ONLY because the attention mask of format
    #             # [0, ..., -10.000, ..., 0, ...,] is not supported
    #             # throw warning
    #             logger.info(
    #                 "Memory efficient attention with `xformers` might currently not work correctly if an attention mask is required for the attention operation."
    #             )
    #             processor = XFormersAttnAddedKVProcessor(attention_op=attention_op)
    #         else:
    #             processor = XFormersAttnProcessor(attention_op=attention_op)
    #     else:
    #         if is_custom_diffusion:
    #             attn_processor_class = (
    #                 CustomDiffusionAttnProcessor2_0
    #                 if hasattr(F, "scaled_dot_product_attention")
    #                 else CustomDiffusionAttnProcessor
    #             )
    #             processor = attn_processor_class(
    #                 train_kv=self.processor.train_kv,
    #                 train_q_out=self.processor.train_q_out,
    #                 hidden_size=self.processor.hidden_size,
    #                 cross_attention_dim=self.processor.cross_attention_dim,
    #             )
    #             processor.load_state_dict(self.processor.state_dict())
    #             if hasattr(self.processor, "to_k_custom_diffusion"):
    #                 processor.to(self.processor.to_k_custom_diffusion.weight.device)
    #         else:
    #             # set attention processor
    #             # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
    #             # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
    #             # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
    #             processor = (
    #                 AttnProcessor2_0()
    #                 if hasattr(F, "scaled_dot_product_attention") and self.scale_qk
    #                 else AttnProcessor()
    #             )

    #     self.set_processor(processor)

    # def set_attention_slice(self, slice_size: int) -> None:
    #     r"""
    #     Set the slice size for attention computation.

    #     Args:
    #         slice_size (`int`):
    #             The slice size for attention computation.
    #     """
    #     pdb.set_trace()

    #     if slice_size is not None and slice_size > self.sliceable_head_dim:
    #         raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

    #     if slice_size is not None and self.added_kv_proj_dim is not None:
    #         processor = SlicedAttnAddedKVProcessor(slice_size)
    #     elif slice_size is not None:
    #         processor = SlicedAttnProcessor(slice_size)
    #     elif self.added_kv_proj_dim is not None:
    #         processor = AttnAddedKVProcessor()
    #     else:
    #         # set attention processor
    #         # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
    #         # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
    #         # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
    #         processor = (
    #             AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
    #         )

    #     self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor") -> None:
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        # ==> pdb.set_trace()

        # if (
        #     hasattr(self, "processor")
        #     and isinstance(self.processor, torch.nn.Module)
        #     and not isinstance(processor, torch.nn.Module)
        # ):
        #     pdb.set_trace()
        #     logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
        #     self._modules.pop("processor")

        self.processor = processor

    # def get_processor(self, return_deprecated_lora: bool = False) -> "AttentionProcessor":
    #     pdb.set_trace()

    #     if not return_deprecated_lora:
    #         return self.processor

    def forward(self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ):
        # encoder_hidden_states = None
        # attention_mask = None
        # cross_attention_kwargs = {'temb': None}

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        # attn_parameters --
        # {'kwargs', 'temb', 'encoder_hidden_states', 'args', 'hidden_states', 'attention_mask', 'attn'}

        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            pdb.set_trace()
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
        # assert cross_attention_kwargs == {}

        return self.processor(self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        pdb.set_trace()

        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    # def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
    #     pdb.set_trace()

    #     head_size = self.heads
    #     if tensor.ndim == 3:
    #         batch_size, seq_len, dim = tensor.shape
    #         extra_dim = 1
    #     else:
    #         batch_size, extra_dim, seq_len, dim = tensor.shape
    #     tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
    #     tensor = tensor.permute(0, 2, 1, 3)

    #     if out_dim == 3:
    #         tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

    #     pdb.set_trace()
    #     return tensor

    # def get_attention_scores(
    #     self, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    # ) -> torch.Tensor:
    #     r"""
    #     Compute the attention scores.
    #     """
    #     pdb.set_trace()

    #     dtype = query.dtype
    #     if self.upcast_attention:
    #         query = query.float()
    #         key = key.float()

    #     if attention_mask is None:
    #         baddbmm_input = torch.empty(
    #             query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
    #         )
    #         beta = 0
    #     else:
    #         baddbmm_input = attention_mask
    #         beta = 1

    #     attention_scores = torch.baddbmm(
    #         baddbmm_input,
    #         query,
    #         key.transpose(-1, -2),
    #         beta=beta,
    #         alpha=self.scale,
    #     )
    #     del baddbmm_input

    #     if self.upcast_softmax:
    #         attention_scores = attention_scores.float()

    #     attention_probs = attention_scores.softmax(dim=-1)
    #     del attention_scores

    #     attention_probs = attention_probs.to(dtype)

    #     return attention_probs

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.
        """
        pdb.set_trace()

        assert attention_mask is None
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    # def norm_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
    #     pdb.set_trace()

    #     assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

    #     if isinstance(self.norm_cross, nn.LayerNorm):
    #         encoder_hidden_states = self.norm_cross(encoder_hidden_states)
    #     elif isinstance(self.norm_cross, nn.GroupNorm):
    #         # Group norm norms along the channels dimension and expects
    #         # input to be in the shape of (N, C, *). In this case, we want
    #         # to norm along the hidden dimension, so we need to move
    #         # (batch_size, sequence_length, hidden_size) ->
    #         # (batch_size, hidden_size, sequence_length)
    #         encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
    #         encoder_hidden_states = self.norm_cross(encoder_hidden_states)
    #         encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
    #     else:
    #         assert False

    #     return encoder_hidden_states

    # @torch.no_grad()
    # def fuse_projections(self, fuse=True):
    #     pdb.set_trace()

    #     device = self.to_q.weight.data.device
    #     dtype = self.to_q.weight.data.dtype

    #     if not self.is_cross_attention:
    #         # fetch weight matrices.
    #         concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
    #         in_features = concatenated_weights.shape[1]
    #         out_features = concatenated_weights.shape[0]

    #         # create a new single projection layer and copy over the weights.
    #         self.to_qkv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
    #         self.to_qkv.weight.copy_(concatenated_weights)
    #         if self.use_bias:
    #             concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
    #             self.to_qkv.bias.copy_(concatenated_bias)

    #     else:
    #         concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
    #         in_features = concatenated_weights.shape[1]
    #         out_features = concatenated_weights.shape[0]

    #         self.to_kv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
    #         self.to_kv.weight.copy_(concatenated_weights)
    #         if self.use_bias:
    #             concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
    #             self.to_kv.bias.copy_(concatenated_bias)

    #     # handle added projections for SD3 and others.
    #     if hasattr(self, "add_q_proj") and hasattr(self, "add_k_proj") and hasattr(self, "add_v_proj"):
    #         concatenated_weights = torch.cat(
    #             [self.add_q_proj.weight.data, self.add_k_proj.weight.data, self.add_v_proj.weight.data]
    #         )
    #         in_features = concatenated_weights.shape[1]
    #         out_features = concatenated_weights.shape[0]

    #         self.to_added_qkv = nn.Linear(
    #             in_features, out_features, bias=self.added_proj_bias, device=device, dtype=dtype
    #         )
    #         self.to_added_qkv.weight.copy_(concatenated_weights)
    #         if self.added_proj_bias:
    #             concatenated_bias = torch.cat(
    #                 [self.add_q_proj.bias.data, self.add_k_proj.bias.data, self.add_v_proj.bias.data]
    #             )
    #             self.to_added_qkv.bias.copy_(concatenated_bias)

    #     self.fused_projections = fuse

class BaseOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    Python dictionary.
    """

    def __init_subclass__(cls) -> None:
        """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output `ModelOutput` subclasses.
        """
        if is_torch_available():
            import torch.utils._pytree

            if is_torch_version("<", "2.2"): # False
                pdb.set_trace()
                torch.utils._pytree._register_pytree_node(
                    cls,
                    torch.utils._pytree._dict_flatten,
                    lambda values, context: cls(**torch.utils._pytree._dict_unflatten(values, context)),
                )
            else:
                torch.utils._pytree.register_pytree_node(
                    cls,
                    torch.utils._pytree._dict_flatten,
                    lambda values, context: cls(**torch.utils._pytree._dict_unflatten(values, context)),
                )

    def __post_init__(self) -> None:
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and isinstance(first_field, dict):
            for key, value in first_field.items():
                self[key] = value
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k: Any) -> Any:
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name: Any, value: Any) -> None:
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> Tuple[Any, ...]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())

# !!! -------------------------------------
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        # tensor [parameters] size: [1, 8, 64, 64], min: -56.005188, max: 35.71368, mean: -9.935047
        # deterministic = False
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # ==> self.mean.size() -- [1, 4, 64, 64]
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        assert self.deterministic == False
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )


    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        # todos.debug.output_var("sample", x)
        return x

    # def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
    #     if self.deterministic:
    #         return torch.Tensor([0.0])
    #     else:
    #         if other is None:
    #             return 0.5 * torch.sum(
    #                 torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
    #                 dim=[1, 2, 3],
    #             )
    #         else:
    #             return 0.5 * torch.sum(
    #                 torch.pow(self.mean - other.mean, 2) / other.var
    #                 + self.var / other.var
    #                 - 1.0
    #                 - self.logvar
    #                 + other.logvar,
    #                 dim=[1, 2, 3],
    #             )
    #     pdb.set_trace()

    # def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
    #     if self.deterministic:
    #         return torch.Tensor([0.0])
    #     logtwopi = np.log(2.0 * np.pi)
    #     return 0.5 * torch.sum(
    #         logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
    #         dim=dims,
    #     )
    #     pdb.set_trace()

    def mode(self) -> torch.Tensor:
        return self.mean

@dataclass
class AutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"  # noqa: F821


@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: "torch.Tensor"  # noqa: F821

@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.Tensor
    commit_loss: Optional[torch.FloatTensor] = None

# --------------------
class AutoencoderKLTemporalDecoder(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types = ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        block_out_channels = [128, 256, 512, 512],
        layers_per_block: int = 2,
        latent_channels: int = 4,
        sample_size: int = 768,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
    ):
        super().__init__()
        # in_channels = 3
        # out_channels = 3
        # down_block_types = ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D']
        # block_out_channels = [128, 256, 512, 512]
        # layers_per_block = 2
        # latent_channels = 4
        # sample_size = 768
        # scaling_factor = 0.18215
        # force_upcast = True
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = TemporalDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)

        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        ) # 768
        # self.config.block_out_channels -- [128, 256, 512, 512]
        self.tile_latent_min_size = int(sample_size / (2 ** (len(block_out_channels) - 1))) # 96
        self.tile_overlap_factor = 0.25
        # pdb.set_trace()

    def _set_gradient_checkpointing(self, module, value=False):
        pdb.set_trace()

        if isinstance(module, (Encoder, TemporalDecoder)):
            module.gradient_checkpointing = value

    # @property
    # # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    # def attn_processors(self) -> Dict[str, AttentionProcessor]:
    #     r"""
    #     Returns:
    #         `dict` of attention processors: A dictionary containing all attention processors used in the model with
    #         indexed by its weight name.
    #     """
    #     # set recursively
    #     processors = {}

    #     def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
    #         if hasattr(module, "get_processor"):
    #             processors[f"{name}.processor"] = module.get_processor()

    #         for sub_name, child in module.named_children():
    #             fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

    #         return processors

    #     for name, module in self.named_children():
    #         fn_recursive_add_processors(name, module, processors)

    #     pdb.set_trace()
    #     return processors

    # # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    # def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
    #     r"""
    #     Sets the attention processor to use to compute attention.
    #     """
    #     pdb.set_trace()

    #     count = len(self.attn_processors.keys())

    #     if isinstance(processor, dict) and len(processor) != count:
    #         raise ValueError(
    #             f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
    #             f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
    #         )

    #     def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
    #         if hasattr(module, "set_processor"):
    #             if not isinstance(processor, dict):
    #                 module.set_processor(processor)
    #             else:
    #                 module.set_processor(processor.pop(f"{name}.processor"))

    #         for sub_name, child in module.named_children():
    #             fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    #     for name, module in self.named_children():
    #         fn_recursive_attn_processor(name, module, processor)

    # def set_default_attn_processor(self):
    #     """
    #     Disables custom attention processors and sets the default attention implementation.
    #     """
    #     if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
    #         processor = AttnProcessor()
    #     else:
    #         raise ValueError(
    #             f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
    #         )

    #     pdb.set_trace()
    #     self.set_attn_processor(processor)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True):
        """
        Encode a batch of images into latents.
        """
        # todos.debug.output_var("encode x", x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        # todos.debug.output_var("encode moments", moments)

        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict: # False
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self,
        z: torch.Tensor,
        num_frames: int,
        return_dict: bool = True,
    ):
        """
        Decode a batch of images.
        """
        # num_frames = 4
        # return_dict = True
        # todos.debug.output_var("decode z", z)

        batch_size = z.shape[0] // num_frames
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=z.dtype, device=z.device)
        decoded = self.decoder(z, num_frames=num_frames, image_only_indicator=image_only_indicator)

        # todos.debug.output_var("decode out", decoded)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        num_frames: int = 1,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        useless ...
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        dec = self.decode(z, num_frames=num_frames).sample

        if not return_dict:
            return (dec,)
        pdb.set_trace()
        return DecoderOutput(sample=dec)

# once !!! ---------------------------------
class UNetMidBlock2D(nn.Module):
    """
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,    
    """
    def __init__(self,
        in_channels = 512,
        temb_channels = None,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_head_dim: int = 512,
        output_scale_factor: float = 1.0,

        attn_groups: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_pre_norm: bool = True,
    ):
        super().__init__()
        # in_channels = 512
        # temb_channels = None
        # dropout = 0.0
        # num_layers = 1
        # resnet_eps = 1e-06
        # resnet_time_scale_shift = 'default'
        # resnet_act_fn = 'silu'
        # resnet_groups = 32
        # attn_groups = 32
        # resnet_pre_norm = True
        # add_attention = True
        # attention_head_dim = 512
        # output_scale_factor = 1
        assert num_layers == 1
        assert resnet_act_fn == 'silu'

        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        attn_groups = resnet_groups

        assert resnet_pre_norm == True
        assert temb_channels == None
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm="default",
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        assert num_layers == 1
        for _ in range(num_layers):
            assert self.add_attention == True
            if self.add_attention: # True
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                    )
                )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm="default",
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        # pdb.set_trace()

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert temb == None
        # todos.debug.output_var("hidden_states1", hidden_states)
        # todos.debug.output_var("temb", temb)

        # (Pdb) self.attentions
        # ModuleList(
        #   (0): Attention(
        #     (group_norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        #     (to_q): Linear(in_features=512, out_features=512, bias=True)
        #     (to_k): Linear(in_features=512, out_features=512, bias=True)
        #     (to_v): Linear(in_features=512, out_features=512, bias=True)
        #     (to_out): ModuleList(
        #       (0): Linear(in_features=512, out_features=512, bias=True)
        #       (1): Dropout(p=0.0, inplace=False)
        #     )
        #   )
        # )
        # (Pdb) self.resnets
        # ModuleList(
        #   (0-1): 2 x ResnetBlock2D(
        #     (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        #     (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        #     (dropout): Dropout(p=0.0, inplace=False)
        #     (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (nonlinearity): SiLU()
        #   )
        # )


        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            # if attn is not None:
            #     hidden_states = attn(hidden_states, temb=temb)
            # else:
            #     pdb.set_trace()
            hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)
        # todos.debug.output_var("hidden_states2", hidden_states)

        return hidden_states

# !!! --------------------------------------------
class TemporalDecoder(nn.Module):
    def __init__(self,
        in_channels: int = 4,
        out_channels: int = 3,
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
    ):
        super().__init__()
        # in_channels = 4
        # out_channels = 3
        # block_out_channels = [128, 256, 512, 512]
        # layers_per_block = 2

        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.mid_block = MidBlockTemporalDecoder(
            num_layers=self.layers_per_block,
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1],
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)): #  block_out_channels -- [128, 256, 512, 512]
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            up_block = UpBlockTemporalDecoder(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-6)

        self.conv_act = nn.SiLU()
        self.conv_out = torch.nn.Conv2d(
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        conv_out_kernel_size = (3, 1, 1)
        padding = [int(k // 2) for k in conv_out_kernel_size] # [1, 0, 0]
        self.time_conv_out = torch.nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=conv_out_kernel_size,
            padding=padding,
        )

        self.gradient_checkpointing = False

    def forward(self,
        sample: torch.Tensor,
        image_only_indicator: torch.Tensor,
        num_frames: int = 1,
    ) -> torch.Tensor:
        # todos.debug.output_var("sample1", sample)
        # todos.debug.output_var("image_only_indicator", image_only_indicator)
        # todos.debug.output_var("num_frames", num_frames)

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        # middle
        sample = self.mid_block(sample, image_only_indicator=image_only_indicator)
        sample = sample.to(upscale_dtype)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample, image_only_indicator=image_only_indicator)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        batch_frames, channels, height, width = sample.shape
        batch_size = batch_frames // num_frames
        sample = sample[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        sample = self.time_conv_out(sample)

        # todos.debug.output_var("sample1", sample)
        sample = sample.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        # todos.debug.output_var("sample3", sample)

        return sample


# !!! --------------------------------------
class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.
    """
    def __init__(self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
    ):
        super().__init__()
        # in_channels = 3
        # out_channels = 4
        # down_block_types = ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D']
        # block_out_channels = [128, 256, 512, 512]
        # layers_per_block = 2
        # norm_num_groups = 32
        # act_fn = 'silu'
        # double_z = True
        # mid_block_add_attention = True

        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            # down_block = get_down_block(
            #     down_block_type,
            #     num_layers=self.layers_per_block,
            #     in_channels=input_channel,
            #     out_channels=output_channel,
            #     add_downsample=not is_final_block,
            #     resnet_eps=1e-6,
            #     downsample_padding=0,
            #     resnet_act_fn=act_fn,
            #     resnet_groups=norm_num_groups,
            #     attention_head_dim=output_channel,
            #     temb_channels=None,
            # )
            down_block = DownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block,
                # dropout=dropout,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                downsample_padding=0,
            )

            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False
        # pdb.set_trace()


    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        # todos.debug.output_var("sample1", sample)
        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # todos.debug.output_var("sample2", sample)
        return sample

# !!! -------------------------------------
class MidBlockTemporalDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_head_dim: int = 512,
        num_layers: int = 1,
        upcast_attention: bool = False,
    ):
        super().__init__()
        # in_channels = 512
        # out_channels = 512
        # attention_head_dim = 512
        # num_layers = 2
        # upcast_attention = False

        resnets = []
        attentions = []
        for i in range(num_layers): # 2
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=1e-6,
                    temporal_eps=1e-5,
                    merge_factor=0.0,
                )
            )

        attentions.append(
            Attention(
                query_dim=in_channels,
                heads=in_channels // attention_head_dim,
                dim_head=attention_head_dim,
                eps=1e-6,
                upcast_attention=upcast_attention,
                norm_num_groups=32,
                bias=True,
                residual_connection=True,
            )
        )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self,
        hidden_states: torch.Tensor,
        image_only_indicator: torch.Tensor,
    ):
        # todos.debug.output_var("hidden_states5", hidden_states)
        # todos.debug.output_var("image_only_indicator", image_only_indicator)

        hidden_states = self.resnets[0](
            hidden_states,
            image_only_indicator=image_only_indicator,
        )
        for resnet, attn in zip(self.resnets[1:], self.attentions):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(
                hidden_states,
                image_only_indicator=image_only_indicator,
            )
        # todos.debug.output_var("hidden_states6", hidden_states)
        return hidden_states

# !!! ------------------------------------
class UpBlockTemporalDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        add_upsample: bool = True,
    ):
        super().__init__()
        # in_channels = 512
        # out_channels = 512
        # num_layers = 3
        # add_upsample = True

        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=1e-6,
                    temporal_eps=1e-5,
                    merge_factor=0.0,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
            
    def forward(self,
        hidden_states: torch.Tensor,
        image_only_indicator: torch.Tensor,
    ) -> torch.Tensor:
        # todos.debug.output_var("hidden_states7", hidden_states)
        # todos.debug.output_var("image_only_indicator", image_only_indicator)

        for resnet in self.resnets:
            hidden_states = resnet(
                hidden_states,
                image_only_indicator=image_only_indicator,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        # todos.debug.output_var("hidden_states8", hidden_states)
        return hidden_states

# --------------------------------------
class DownEncoderBlock2D(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        # in_channels = 128
        # out_channels = 128
        # dropout = 0.0
        # num_layers = 2
        # resnet_eps = 1e-06
        # resnet_time_scale_shift = 'default'
        # resnet_act_fn = 'silu'
        # resnet_groups = 32
        # resnet_pre_norm = True
        # output_scale_factor = 1.0
        # add_downsample = True
        # downsample_padding = 0        
                
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm="default",
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None
        # pdb.set_trace()

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


# !!! ----------------------------
class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.
    """
    def __init__(self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        super().__init__()
        # channels = 128
        # use_conv = True
        # out_channels = 128
        # padding = 0
        # name = 'op'
        # kernel_size = 3
        # norm_type = None
        # eps = None
        # elementwise_affine = None
        # bias = True

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        self.norm = None

        if use_conv:
            conv = nn.Conv2d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv
        # pdb.set_trace()

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels

        assert self.use_conv and self.padding == 0
        if self.use_conv and self.padding == 0: # True
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


# !!! ---------------------------------------
class Upsample2D(nn.Module):
    """A 2D upsampling layer with an optional convolution.
    """
    def __init__(self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        # channels = 512
        # use_conv = True
        # use_conv_transpose = False
        # out_channels = 512
        # name = 'conv'
        # kernel_size = 3
        # padding = 1
        # norm_type = None
        # eps = None
        # elementwise_affine = None
        # bias = True
        # interpolate = True
        
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        self.norm = None

        assert use_conv_transpose == False
        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv
        # pdb.set_trace()


    def forward(self, hidden_states: torch.Tensor, output_size= None):
        assert hidden_states.shape[1] == self.channels

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate: # True
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv: # True
            if self.name == "conv": # True
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states

# !!! ---------------------------------
class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.
    """
    def __init__(self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift,
        kernel: Optional[torch.Tensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        # in_channels = 128
        # out_channels = 128
        # conv_shortcut = False
        # dropout = 0.0
        # temb_channels = None
        # groups = 32
        # groups_out = 32
        # pre_norm = True
        # eps = 1e-06
        # non_linearity = 'silu'
        # skip_time_act = False
        # time_embedding_norm = 'default'
        # kernel = None
        # output_scale_factor = 1.0
        # use_in_shortcut = None
        # up = False
        # down = False
        # conv_shortcut_bias = True
        # conv_2d_out_channels = 128
        assert time_embedding_norm == 'default'

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None

        assert kernel != "fir" and kernel != "sde_vp"
        if self.up: # False
            self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down: # False
            self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )
        assert self.time_emb_proj == None

    # def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:

    def forward(self, input_tensor, temb):
        # temb = None
        # args = ()
        # kwargs = {}

        # if len(args) > 0 or kwargs.get("scale", None) is not None:
        #     deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        #     deprecate("scale", "1.0.0", deprecation_message)

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        assert self.upsample == None
        assert self.downsample == None

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)


        # self.time_embedding_norm -- 'default'
        assert self.time_embedding_norm == 'default'

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        # assert self.conv_shortcut == None
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


# !!! ---------------------------------
class TemporalResnetBlock(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
    ):
        super().__init__()
        # in_channels = 512
        # out_channels = 512
        # temb_channels = None
        # eps = 1e-05
        
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        kernel_size = (3, 1, 1)
        padding = [k // 2 for k in kernel_size]

        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        assert temb_channels == None
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(0.0)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.nonlinearity = get_activation("silu")

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(in_channels, out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        # pdb.set_trace()

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, :, None, None]
            temb = temb.permute(0, 2, 1, 3, 4)
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor

# !!! ------------------------------------------
# VideoResBlock
class SpatioTemporalResBlock(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        temporal_eps: Optional[float] = None,
        merge_factor: float = 0.5,
    ):
        super().__init__()
        # in_channels = 512
        # out_channels = 512
        # temb_channels = None
        # eps = 1e-06
        # temporal_eps = 1e-05
        # merge_factor = 0.0

        self.spatial_res_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=eps,
        )

        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels if out_channels is not None else in_channels,
            out_channels=out_channels if out_channels is not None else in_channels,
            temb_channels=temb_channels,
            eps=temporal_eps if temporal_eps is not None else eps,
        )

        self.time_mixer = AlphaBlender(alpha=merge_factor)

    def forward(self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ):
        num_frames = image_only_indicator.shape[-1]
        hidden_states = self.spatial_res_block(hidden_states, temb)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )

        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)

        hidden_states = self.temporal_res_block(hidden_states, temb)
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        return hidden_states

# !!! -------------------------------
class AlphaBlender(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))

    def forward(self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ):
        alpha = torch.sigmoid(self.mix_factor)
        alpha = alpha.to(x_spatial.dtype)
        alpha = 1.0 - alpha

        x = alpha * x_spatial + (1.0 - alpha) * x_temporal
        return x

class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            # ==> pdb.set_trace()
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            pdb.set_trace()
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            # ==> pdb.set_trace()
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # if attn.norm_q is not None:
        #     pdb.set_trace()
        #     query = attn.norm_q(query)
        # if attn.norm_k is not None:
        #     pdb.set_trace()
        #     key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            # ==> pdb.set_trace()
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        # else:
        #     pdb.set_trace()

        # assert attn.rescale_output_factor == 1.0
        # hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


if __name__ == "__main__":
    # args.pretrained_model_name_or_path === 'checkpoints/SVD/stable-video-diffusion-img2vid-xt'
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        'checkpoints/SVD/stable-video-diffusion-img2vid-xt', subfolder="vae", revision=None)
    print(vae)
