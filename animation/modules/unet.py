from typing import Dict, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin



from diffusers.models.modeling_utils import ModelMixin

from animation.modules.unet_3d_blocks import get_down_block, UNetMidBlockSpatioTemporal, get_up_block
import pdb
import todos

# from diffusers.utils import logging
# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):

    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        # if cond_proj_dim is not None:
        #     self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        # else:
        #     self.cond_proj = None

        self.act = nn.SiLU() # get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        # if post_act_fn is None:
        #     self.post_act = None
        # else:
        #     self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        # if condition is not None:
        #     sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        # if self.post_act is not None:
        #     sample = self.post_act(sample)
        return sample


class UNetSpatioTemporalConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    A conditional Spatio-Temporal UNet model that takes a noisy video frames, conditional state,
    and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            sample_size = 96,
            in_channels = 8,
            out_channels = 4,
            down_block_types = (
                    "CrossAttnDownBlockSpatioTemporal",
                    "CrossAttnDownBlockSpatioTemporal",
                    "CrossAttnDownBlockSpatioTemporal",
                    "DownBlockSpatioTemporal",
            ),
            up_block_types = (
                    "UpBlockSpatioTemporal",
                    "CrossAttnUpBlockSpatioTemporal",
                    "CrossAttnUpBlockSpatioTemporal",
                    "CrossAttnUpBlockSpatioTemporal",
            ),
            block_out_channels  = (320, 640, 1280, 1280),
            addition_time_embed_dim = 256,
            projection_class_embeddings_input_dim = 768,
            layers_per_block = [2, 2, 2, 2],
            cross_attention_dim = (1024, 1024, 1024, 1024),
            transformer_layers_per_block = [1, 1, 1, 1],
            num_attention_heads  = (5, 10, 20, 20),
            num_frames = 25,
    ):
        super().__init__()
        assert len(down_block_types) == 4
        assert len(up_block_types) == 4

        # input
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0], # 320
            kernel_size=3,
            padding=1,
        )

        # time
        time_embed_dim = block_out_channels[0] * 4 # 1280

        self.time_proj = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0] # 320

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.add_time_proj = Timesteps(addition_time_embed_dim, True, downscale_freq_shift=0)
        self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # cross_attention_dim === 1024
        cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        # layers_per_block === 2
        layers_per_block = [layers_per_block] * len(down_block_types) # len(down_block_types) === 4

        # transformer_layers_per_block === 1
        transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        # # down_block_types --
        #     ['CrossAttnDownBlockSpatioTemporal', 'CrossAttnDownBlockSpatioTemporal', 
        #     'CrossAttnDownBlockSpatioTemporal', 'DownBlockSpatioTemporal']

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                resnet_act_fn="silu",
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        # up_block_types ---
        #     ['UpBlockSpatioTemporal', 'CrossAttnUpBlockSpatioTemporal', 
        #     'CrossAttnUpBlockSpatioTemporal', 'CrossAttnUpBlockSpatioTemporal']

        for i, up_block_type in enumerate(up_block_types):
            # is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            # if not is_final_block:
            #     add_upsample = True
            # else:
            #     add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1, # reversed_layers_per_block -- [2, 2, 2, 2]
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=(i != len(block_out_channels) - 1), #add_upsample,
                resnet_eps=1e-5,
                # resolution_idx=i,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                resnet_act_fn="silu",
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
        # self.conv_norm_out -- GroupNorm(32, 320, eps=1e-05, affine=True)
        self.conv_act = nn.SiLU()

        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )
        # self.conv_out -- Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    @property
    def attn_processors(self):
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
                name: str,
                module: torch.nn.Module,
                processors,
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.
        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            added_time_ids: torch.Tensor,
            pose_latents: torch.Tensor = None,
            image_only_indicator: bool = False,
            return_dict: bool = True,
    ):
        # pose_latents = None
        # image_only_indicator = False
        # return_dict = False
        # tensor [sample] size: [1, 16, 8, 64, 64], min: -4.78125, max: 4.757812, mean: 0.000227
        # timestep --- tensor(1.637770, device='cuda:0')
        # tensor [encoder_hidden_states] size: [1, 5, 1024], min: 0.0, max: 0.0, mean: 0.0
        # tensor [added_time_ids] size: [1, 3], min: 0.020004, max: 127.0, mean: 44.340004

        # 1. time
        timesteps = timestep
        # if not torch.is_tensor(timesteps): # False
        #     pdb.set_trace()
        #     # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        #     # This would be a good case for the `match` statement (Python 3.10+)
        #     is_mps = sample.device.type == "mps"
        #     if isinstance(timestep, float):
        #         dtype = torch.float32 if is_mps else torch.float64
        #     else:
        #         dtype = torch.int32 if is_mps else torch.int64
        #     timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        # elif len(timesteps.shape) == 0: # True
        #     # ==> pdb.set_trace()
        #     timesteps = timesteps[None].to(sample.device)
        if len(timesteps.shape) == 0: # True
            # ==> pdb.set_trace()
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.conv_in(sample)
        if pose_latents is not None:
            # ==> pdb.set_trace()
            sample = sample + pose_latents
        else:
            # ==> pdb.set_trace() for pose_latents == None
            pass

        # assert image_only_indicator == False
        # image_only_indicator = torch.ones(batch_size, num_frames, dtype=sample.dtype, device=sample.device) \
        #     if image_only_indicator else torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # downsample_block -- CrossAttnDownBlockSpatioTemporal
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                # downsample_block -- DownBlockSpatioTemporal
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # 5. up
        # down_block_res_samples:  is tuple: len = 12
        #     tensor [item] size: [16, 320, 64, 64], min: -13.421875, max: 10.726562, mean: 0.005008
        #     tensor [item] size: [16, 320, 64, 64], min: -11.304688, max: 7.28125, mean: -0.061166
        #     tensor [item] size: [16, 320, 64, 64], min: -10.546875, max: 9.632812, mean: -0.003734
        #     tensor [item] size: [16, 320, 32, 32], min: -22.453125, max: 21.15625, mean: 0.002473
        #     tensor [item] size: [16, 640, 32, 32], min: -11.367188, max: 12.890625, mean: -0.052997
        #     tensor [item] size: [16, 640, 32, 32], min: -11.203125, max: 13.078125, mean: -0.013832
        #     tensor [item] size: [16, 640, 16, 16], min: -28.921875, max: 35.625, mean: -0.07044
        #     tensor [item] size: [16, 1280, 16, 16], min: -28.765625, max: 23.71875, mean: -0.098765
        #     tensor [item] size: [16, 1280, 16, 16], min: -23.140625, max: 23.734375, mean: -0.103224
        #     tensor [item] size: [16, 1280, 8, 8], min: -39.1875, max: 38.34375, mean: -0.234854
        #     tensor [item] size: [16, 1280, 8, 8], min: -39.6875, max: 33.15625, mean: -0.312641
        #     tensor [item] size: [16, 1280, 8, 8], min: -42.125, max: 32.71875, mean: -0.280794

        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                # upsample_block -- CrossAttnUpBlockSpatioTemporal
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                # upsample_block -- UpBlockSpatioTemporal
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        return (sample,)
