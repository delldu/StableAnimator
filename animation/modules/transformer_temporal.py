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
# from dataclasses import dataclass

import inspect
from typing import Any, Dict, Optional, Callable

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from animation.modules.attention_processor import AnimationAttnProcessor

import pdb
import todos

class AlphaBlender(nn.Module):
    r"""
    A module to blend spatial and temporal features.

    """

    def __init__(
        self,
        alpha: float,
    ):
        super().__init__()
        self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))

    def get_alpha(self, image_only_indicator: torch.Tensor, ndims: int) -> torch.Tensor:
        alpha = torch.where(
            image_only_indicator.bool(),
            torch.ones(1, 1, device=image_only_indicator.device),
            torch.sigmoid(self.mix_factor)[..., None],
        )

        # (batch, channel, frames, height, width)
        alpha = alpha.reshape(-1)[:, None, None]
        return alpha

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # tensor [x_spatial] size: [16, 4096, 320], min: -2.132812, max: 1.702148, mean: -0.039416
        # tensor [x_temporal] size: [16, 4096, 320], min: -4.917969, max: 5.1875, mean: -0.088254


        alpha = self.get_alpha(image_only_indicator, x_spatial.ndim)
        alpha = alpha.to(x_spatial.dtype)

        x = alpha * x_spatial + (1.0 - alpha) * x_temporal
        # tensor [x] size: [16, 4096, 320], min: -2.710938, max: 3.162109, mean: -0.063549

        return x

class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.
        inner_dim,
        num_attention_heads,
        attention_head_dim,
        cross_attention_dim=cross_attention_dim,

    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim  = 1024,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        # only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.double_self_attention = double_self_attention
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        # self.only_cross_attention = only_cross_attention


        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm


        self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None, #cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim if not double_self_attention else None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )  # is self-attn if encoder_hidden_states is none

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Fuser

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]
        norm_hidden_states = self.norm1(hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None, #encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            # ==> pdb.set_trace()
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            # if self.pos_embed is not None and self.norm_type != "ada_norm_single":
            #     norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class TransformerSpatioTemporalModel(nn.Module):
    """
    A Transformer model for video-like data.
    """

    def __init__(
        self,
        num_attention_heads: int = 5,
        attention_head_dim: int = 64,
        in_channels: int = 320,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = 1024,
        num_tokens=4,
    ):
        super().__init__()
        # print("cross_attention_dim1: ", cross_attention_dim)

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        # assert self.inner_dim == 320

        # 2. Define input layers
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for d in range(num_layers)
            ]
        )

        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(
                    inner_dim,
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        time_embed_dim = in_channels * 4
        self.time_pos_embed = TimestepEmbedding(in_channels, time_embed_dim, out_dim=in_channels)
        # (Pdb) self.time_pos_embed
        # TimestepEmbedding(
        #   (linear_1): Linear(in_features=320, out_features=1280, bias=True)
        #   (act): SiLU()
        #   (linear_2): Linear(in_features=1280, out_features=320, bias=True)
        # )
        self.time_proj = Timesteps(in_channels, True, 0)
        self.time_mixer = AlphaBlender(alpha=0.5)

        # 4. Define output layers
        self.proj_out = nn.Linear(inner_dim, in_channels)
        # self.proj_out -- Linear(in_features=320, out_features=320, bias=True)

        self.gradient_checkpointing = False
        self.num_tokens = num_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # 1. Input
        batch_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames


        end_pos = encoder_hidden_states.shape[1] - self.num_tokens
        time_context = encoder_hidden_states[:, :end_pos, :]

        time_context_first_timestep = time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0]
        time_context = time_context_first_timestep[:, None].broadcast_to(
            batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
        )
        time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1])

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # 2. Blocks
        for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

            hidden_states_mix = hidden_states
            hidden_states_mix = hidden_states_mix + emb

            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        return (output,)


class Attention(nn.Module):
    r"""
    A cross attention layer.

    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None, # 1024 or None
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        scale_qk: bool = True,
        # only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        pre_only=False,
    ):
        super().__init__()

        # To prevent circular import.
        # from .normalization import FP32LayerNorm, RMSNorm

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.pre_only = pre_only

        self.scale_qk = scale_qk

        assert self.scale_qk == True
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        # self.only_cross_attention = only_cross_attention

        # if self.added_kv_proj_dim is None and self.only_cross_attention:
        #     raise ValueError(
        #         "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
        #     )

        self.group_norm = None
        self.spatial_norm = None
        self.norm_q = None
        self.norm_k = None

        # cross_attention_dim == 1024 or None
        assert cross_attention_norm == None

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        # if not self.only_cross_attention:
        #     # only relevant for the `AddedKVProcessor` classes
        #     self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        #     self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        # else:
        #     pdb.set_trace()
        #     self.to_k = None
        #     self.to_v = None
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)


        self.added_proj_bias = added_proj_bias
        # if self.added_kv_proj_dim is not None:
        #     self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
        #     self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
        #     # if self.context_pre_only is not None:
        #     #     self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))
        else:
            pdb.set_trace()


        self.norm_added_q = None
        self.norm_added_k = None

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None:
            # ==> pdb.set_trace()
            assert hasattr(F, "scaled_dot_product_attention") and self.scale_qk
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
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
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor

    def get_processor(self, return_deprecated_lora: bool = False) -> "AttentionProcessor":
        r"""
        Get the attention processor in use.

        Args:
            return_deprecated_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to return the deprecated LoRA attention processor.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        if not return_deprecated_lora:
            return self.processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.



        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

    def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        pdb.set_trace()

        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        """
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

    def norm_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`torch.Tensor`): Hidden states of the encoder.

        Returns:
            `torch.Tensor`: The normalized encoder hidden states.
        """
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states

    # @torch.no_grad()
    # def fuse_projections(self, fuse=True):
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

class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        pdb.set_trace()

        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # if attention_mask is not None:
        #     pdb.set_trace()
        #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     # scaled_dot_product_attention expects attention_mask shape to be
        #     # (batch, heads, source_length, target_length)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # if attn.group_norm is not None:
        #     pdb.set_trace()
        #     hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # elif attn.norm_cross:
        #     pdb.set_trace()
        #     encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

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
            pdb.set_trace()
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        assert attn.residual_connection == False
        # if attn.residual_connection:
        #     pdb.set_trace()
        #     hidden_states = hidden_states + residual

        # assert attn.rescale_output_factor == 1.0
        # hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class FeedForward(nn.Module):
    r"""
    A feed-forward layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class GEGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate)

        pdb.set_trace()
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states, *args, **kwargs):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        hidden_states = self.proj(hidden_states)
        # if is_torch_npu_available():
        #     # using torch_npu.npu_geglu can run faster and save memory on NPU.
        #     return torch_npu.npu_geglu(hidden_states, dim=-1, approximate=1)[0]
        # else:
        #     hidden_states, gate = hidden_states.chunk(2, dim=-1)
        #     return hidden_states * self.gelu(gate)

        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)

class TemporalBasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block for video like data.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        time_mix_inner_dim (`int`): The number of channels for temporal attention.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.

        inner_dim,
        inner_dim,
        num_attention_heads,
        attention_head_dim,
        cross_attention_dim=cross_attention_dim,
    """

    def __init__(
        self,
        dim: int,
        time_mix_inner_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim = 1024,
    ):
        super().__init__()
        assert cross_attention_dim is not None

        self.is_res = dim == time_mix_inner_dim

        self.norm_in = nn.LayerNorm(dim)

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.ff_in = FeedForward(
            dim,
            dim_out=time_mix_inner_dim,
            activation_fn="geglu",
        )

        self.norm1 = nn.LayerNorm(time_mix_inner_dim)
        self.attn1 = Attention(
            query_dim=time_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(time_mix_inner_dim)
        self.attn2 = Attention(
            query_dim=time_mix_inner_dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
        )  # is self-attn if encoder_hidden_states is none

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(time_mix_inner_dim)
        self.ff = FeedForward(time_mix_inner_dim, activation_fn="geglu")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        # chunk dim should be hardcoded to 1 to have better speed vs. memory trade-off
        self._chunk_dim = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        batch_frames, seq_length, channels = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * seq_length, num_frames, channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            pdb.set_trace()
            hidden_states = _chunked_feed_forward(self.ff_in, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            pdb.set_trace()

            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * num_frames, seq_length, channels)

        return hidden_states
