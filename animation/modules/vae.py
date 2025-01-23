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

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import is_torch_version, logging
from diffusers.utils.import_utils import is_torch_available


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


import pdb
import todos

# ------------------------------
class Attention(nn.Module):
    def __init__(self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = True,
        upcast_softmax: bool = False,
        norm_num_groups = None,
        spatial_norm_dim = None,
        eps: float = 1e-6,
        residual_connection: bool = True,

        processor = None,
    ):
        super().__init__()

        # To prevent circular import.
        # from .normalization import FP32LayerNorm, RMSNorm
        assert processor == None

        self.heads = heads
        self.inner_dim = dim_head * heads
        self.cross_attention_dim = query_dim
        self.upcast_softmax = upcast_softmax
        self.residual_connection = residual_connection # !!!
        self.out_dim = query_dim
        self.scale = dim_head**-0.5

        if norm_num_groups is not None: # True | False
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        self.spatial_norm = None # !!!
        self.norm_q = None
        self.norm_k = None
        self.norm_cross = None


        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        # only relevant for the `AddedKVProcessor` classes
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=True))
        self.to_out.append(nn.Dropout(0.0))

        if processor is None:
            processor = AttnProcessor2_0()
        self.set_processor(processor)

    def set_processor(self, processor):
        self.processor = processor

    def forward(self,
        hidden_states: torch.Tensor,
        encoder_hidden_states = None,
        attention_mask = None,
        **cross_attention_kwargs,
    ):
        # encoder_hidden_states = None
        # attention_mask = None
        # cross_attention_kwargs = {'temb': None}

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        # attn_parameters --
        # {'kwargs', 'temb', 'encoder_hidden_states', 'args', 'hidden_states', 'attention_mask', 'attn'}
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
        # assert cross_attention_kwargs == {}

        return self.processor(self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )


class BaseOutput(OrderedDict):
    def __init_subclass__(cls) -> None:
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

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)


    def to_tuple(self):
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


    def sample(self, generator = None):
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

    def mode(self) -> torch.Tensor:
        return self.mean

class AutoencoderKLOutput(BaseOutput):
    latent_dist: "DiagonalGaussianDistribution"  # noqa: F821


class Transformer2DModelOutput(BaseOutput):
    sample: "torch.Tensor"  # noqa: F821

class DecoderOutput(BaseOutput):
    sample: torch.Tensor
    commit_loss = None

# --------------------
class AutoencoderKLTemporalDecoder(ModelMixin, ConfigMixin):
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
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
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

    # @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True):
        # todos.debug.output_var("encode x", x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        # todos.debug.output_var("encode moments", moments)

        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict: # False
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    # @apply_forward_hook
    def decode(self,
        z: torch.Tensor,
        num_frames: int,
        return_dict: bool = True,
    ):
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
        generator = None,
        num_frames: int = 1,
    ):
        r""" useless ... """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        dec = self.decode(z, num_frames=num_frames).sample

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

# once !!! ---------------------------------
class UNetMidBlock2D(nn.Module):
    def __init__(self,
        in_channels = 512,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        attention_head_dim: int = 512,
        num_layers: int = 1,
    ):
        super().__init__()
        # in_channels = 512
        # num_layers = 1
        # resnet_eps = 1e-06
        # resnet_groups = 32
        # attention_head_dim = 512
        assert num_layers == 1
        assert num_layers == 1

        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                non_linearity="silu",
                pre_norm=True,
            )
        ]
        attentions = []

        assert num_layers == 1
        for _ in range(num_layers):
            attentions.append(
                Attention(
                    in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
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
                    eps=resnet_eps,
                    groups=resnet_groups,
                    non_linearity="silu",
                    pre_norm=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor, temb = None):
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
        #     (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (nonlinearity): SiLU()
        #   )
        # )
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)
        # todos.debug.output_var("hidden_states2", hidden_states)

        return hidden_states

# !!! once --------------------------------------------
class TemporalDecoder(nn.Module):
    def __init__(self,
        in_channels: int = 4,
        out_channels: int = 3,
        block_out_channels = (128, 256, 512, 512),
        layers_per_block: int = 2,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.mid_block = MidBlockTemporalDecoder(
            num_layers=layers_per_block,
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
                num_layers=layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-6)

        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        conv_out_kernel_size = (3, 1, 1)
        padding = [int(k // 2) for k in conv_out_kernel_size] # [1, 0, 0]
        self.time_conv_out = nn.Conv3d(
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


# !!! once --------------------------------------
class Encoder(nn.Module):
    def __init__(self,
        in_channels: int = 3,
        out_channels: int = 4,
        down_block_types = ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        block_out_channels = [128, 256, 512, 512],
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
    ):
        super().__init__()
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

            down_block = DownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_groups=norm_num_groups,
                downsample_padding=0,
            )

            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False


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

# !!! once -------------------------------------
class MidBlockTemporalDecoder(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        attention_head_dim: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        # in_channels = 512
        # out_channels = 512
        # attention_head_dim = 512
        # num_layers = 2

        resnets = []
        attentions = []
        assert num_layers == 2
        for i in range(num_layers): # 2
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=input_channels,
                    out_channels=out_channels,
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
    def __init__(self,
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
        in_channels,
        out_channels,
        num_layers = 1,
        resnet_eps = 1e-6,
        resnet_groups = 32,
        add_downsample = True,
        downsample_padding = 0,
    ):
        super().__init__()
        # in_channels = 128
        # out_channels = 128
        # num_layers = 2
        # resnet_eps = 1e-06
        # resnet_groups = 32
        # add_downsample = True
        # downsample_padding = 0        
                
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    non_linearity="silu",
                    pre_norm=True,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [ Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding) ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


# !!! ----------------------------
class Downsample2D(nn.Module):
    def __init__(self,
        channels: int,
        use_conv: bool = True,
        out_channels = None,
        padding: int = 1,
    ):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, stride=2, padding=padding, bias=True)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=2, stride=2)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels

        pad = (0, 1, 0, 1)
        hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


# !!! ---------------------------------------
class Upsample2D(nn.Module):
    def __init__(self,
        channels: int,
        use_conv: bool = False,
        out_channels = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1, bias=True)
        else:
            conv = None

        self.conv = conv


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
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv: # True
            hidden_states = self.conv(hidden_states)

        return hidden_states

# !!! ---------------------------------
class ResnetBlock2D(nn.Module):
    r"""
    """
    def __init__(self,
        *,
        in_channels: int,
        out_channels = None,
        groups: int = 32,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "silu",

        conv_shortcut: bool = False,
    ):
        super().__init__()
        assert conv_shortcut == False


        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU() # get_activation(non_linearity)
        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        # assert temb == None
        # if temb is not None:
        #     hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        # assert self.conv_shortcut == None
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states)

        return output_tensor


# !!! ---------------------------------
class TemporalResnetBlock(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        # in_channels = 512
        # out_channels = 512
        # eps = 1e-05
        
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        kernel_size = (3, 1, 1)
        padding = [k // 2 for k in kernel_size]

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps, affine=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.nonlinearity = nn.SiLU() # get_activation("silu")

        self.use_in_shortcut = self.in_channels != out_channels
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(in_channels, out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.conv_shortcut = None


    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor

# !!! ------------------------------------------
# VideoResBlock
class SpatioTemporalResBlock(nn.Module):
    def __init__(self,
        in_channels = 512,
        out_channels  = 512,
        eps = 1e-6,
        temporal_eps = 1e-5,
        merge_factor = 0.0,
    ):
        super().__init__()
        self.spatial_res_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            eps=eps,
        )

        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            eps=temporal_eps,
        )

        self.time_mixer = AlphaBlender(alpha=merge_factor)

    def forward(self,
        hidden_states,
        temb = None,
        image_only_indicator = None,
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
        self.register_parameter("mix_factor", nn.Parameter(torch.Tensor([alpha])))

    def forward(self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator = None,
    ):
        alpha = torch.sigmoid(self.mix_factor)
        alpha = alpha.to(x_spatial.dtype)
        alpha = 1.0 - alpha

        x = alpha * x_spatial + (1.0 - alpha) * x_temporal
        return x

# xxxx_debug
class AttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states = None,
        attention_mask = None,
        temb = None,
    ):
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            # ==> pdb.set_trace()
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # assert  attention_mask is not None
        if attention_mask is not None:
            # ==> pdb.set_trace()
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
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

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        # hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            # ==> pdb.set_trace()
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection: # True
            hidden_states = hidden_states + residual

        return hidden_states


if __name__ == "__main__":
    # args.pretrained_model_name_or_path === 'checkpoints/SVD/stable-video-diffusion-img2vid-xt'
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        'checkpoints/SVD/stable-video-diffusion-img2vid-xt', subfolder="vae", revision=None)
    print(vae)
