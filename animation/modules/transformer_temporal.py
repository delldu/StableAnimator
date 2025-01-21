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
# from animation.modules.unet import TimestepEmbedding, Timesteps

import pdb
import todos

class AlphaBlender(nn.Module):
    r"""
    A module to blend spatial and temporal features.
    """
    def __init__(self, alpha: float):
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

        # tensor [torch.sigmoid(self.mix_factor)[..., None]] size: [1, 1], min: 0.593262, max: 0.593262, mean: 0.593262
        # tensor [alpha] size: [16, 1, 1], min: 0.593262, max: 0.593262, mean: 0.593262
        return alpha

    def forward(self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ):
        # tensor [x_spatial] size: [16, 4096, 320], min: -2.132812, max: 1.702148, mean: -0.039416
        # tensor [x_temporal] size: [16, 4096, 320], min: -4.917969, max: 5.1875, mean: -0.088254
        # tensor [AlphaBlender image_only_indicator] size: [1, 16], min: 0.0, max: 0.0, mean: 0.0

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
    def __init__(self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim  = 1024,
    ):
        super().__init__()
        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-5)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,
            cross_attention_dim=None,
        )

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(dim, 1e-5, elementwise_affine=True)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim, 
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,
        )  # is self-attn if encoder_hidden_states is none

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, 1e-5, elementwise_affine=True)
        self.ff = FeedForward(dim)


    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # encoder_attention_mask = None
        # timestep = None

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        batch_size = hidden_states.shape[0]
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None, #encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
        )

        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
        )
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states

        return hidden_states


class TransformerSpatioTemporalModel(nn.Module):
    """ A Transformer model for video-like data. """
    def __init__(self,
        num_attention_heads: int = 5,
        attention_head_dim: int = 64,
        in_channels: int = 320,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = 1024,
        num_tokens=4,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

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

        self.time_pos_embed = TimestepEmbedding(in_channels, in_channels * 4, out_dim=in_channels)
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

    def forward(self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        # tensor [hidden_states] size: [16, 320, 64, 64], min: -17.109375, max: 39.46875, mean: 0.010112
        # tensor [encoder_hidden_states] size: [16, 5, 1024], min: -14.492188, max: 14.453125, mean: 0.000888
        # tensor [image_only_indicator] size: [1, 16], min: 0.0, max: 0.0, mean: 0.0

        assert return_dict == False

        # 1. Input
        batch_frames, _, height, width = hidden_states.shape # size: [16, 320, 64, 64]
        num_frames = image_only_indicator.shape[-1] # size: [1, 16]
        batch_size = batch_frames // num_frames # ==> 1

        end_pos = encoder_hidden_states.shape[1] - self.num_tokens # ==> 1
        time_context = encoder_hidden_states[:, :end_pos, :]
        # tensor [time_context1] size: [16, 1, 1024], min: -5.863281, max: 6.507812, mean: 0.004285

        time_context_first_timestep = time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0]
        # tensor [time_context_first_timestep] size: [1, 1, 1024], min: -5.863281, max: 6.507812, mean: 0.004285

        time_context = time_context_first_timestep[:, None].broadcast_to(
            batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
        )
        # tensor [time_context] size: [1, 4096, 1, 1024], min: -5.863281, max: 6.507812, mean: 0.004285
        time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1])
        # tensor [time_context] size: [4096, 1, 1024], min: -5.863281, max: 6.507812, mean: 0.004285

        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]

        # tensor [hidden_states1] size: [16, 320, 64, 64], min: -2.525391, max: 2.4375, mean: -0.013387
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
        # tensor [hidden_states2] size: [16, 4096, 320], min: -2.525391, max: 2.4375, mean: -0.013387

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
        # tensor [hidden_states3] size: [16, 4096, 320], min: -28.609375, max: 41.75, mean: -1.081215
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        # tensor [hidden_states4] size: [16, 320, 64, 64], min: -28.609375, max: 41.75, mean: -1.081215

        output = hidden_states + residual
        # tensor [output] size: [16, 320, 64, 64], min: -35.4375, max: 79.375, mean: -1.071102

        return (output,)


class Attention(nn.Module):
    r"""
    A cross attention layer.
        query_dim=time_mix_inner_dim,
        heads=num_attention_heads,
        dim_head=attention_head_dim,
        cross_attention_dim=None,

    """
    def __init__(self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None, # 1024 or None
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        scale_qk: bool = True,
        processor: Optional["AttnProcessor"] = None,
    ):
        super().__init__()
        # assert processor == None

        self.inner_dim = dim_head * heads
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.rescale_output_factor = 1.0
        self.residual_connection = False
        self.out_dim = query_dim

        self.scale_qk = scale_qk

        assert self.scale_qk == True
        self.scale = dim_head**-0.5

        self.heads =  heads

        self.group_norm = None
        self.spatial_norm = None

        # cross_attention_dim == 1024 or None
        self.norm_cross = None
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=False)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=True))
        self.to_out.append(nn.Dropout(dropout))

        # set attention processor
        self.set_processor(processor)


    def set_processor(self, processor: "AttnProcessor") -> None:
        self.processor = processor

    def get_processor(self, return_deprecated_lora: bool = False) -> "AttentionProcessor":
        r"""
        Get the attention processor in use.
        """
        assert return_deprecated_lora == True
        if not return_deprecated_lora: # False
            return self.processor

    def forward(self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # **cross_attention_kwargs,
    ) -> torch.Tensor:
        #
        # encoder_hidden_states = None
        # attention_mask = None
        # cross_attention_kwargs = {}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.
        """
        # self.heads == 5
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.
        """
        assert out_dim == 3
        assert tensor.ndim == 3

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
        pdb.set_trace()

        dtype = query.dtype

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

        # if self.upcast_softmax:
        #     attention_scores = attention_scores.float()

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
        # head_size = self.heads
        assert attention_mask == None
        return attention_mask


class FeedForward(nn.Module):
    def __init__(self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        act_fn = GEGLU(dim, inner_dim, bias=True)
        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out, bias=True))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        # (Pdb) self
        # FeedForward(
        #   (net): ModuleList(
        #     (0): GEGLU(
        #       (proj): Linear(in_features=320, out_features=2560, bias=True)
        #     )
        #     (1): Dropout(p=0.0, inplace=False)
        #     (2): Linear(in_features=1280, out_features=320, bias=True)
        #   )
        # )


    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # args = ()
        # kwargs = {}
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class GEGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.
    """
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, hidden_states, *args, **kwargs):
        # args = ()
        # kwargs = {}
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)

class TemporalBasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block for video like data.

    Parameters:

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
        self.ff_in = FeedForward(dim, dim_out=time_mix_inner_dim)

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
        self.ff = FeedForward(time_mix_inner_dim)

    def forward(self,
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
        hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states

        # 3. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * num_frames, seq_length, channels)

        return hidden_states
