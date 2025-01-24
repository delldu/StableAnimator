import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .vae import (
#     # Downsample2D,
#     # SpatioTemporalResBlock,
#     # ResnetBlock2D,
#     Upsample2D,
#     # TemporalResnetBlock,
# )
import pdb
import todos
from diffusers.utils.import_utils import is_xformers_available

# xxxx_debug
if is_xformers_available():
    import xformers
    from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
else:
    print(1 / 0)


class UNetSpatioTemporalConditionModel(nn.Module):
    def __init__(
        self,
        sample_size=96,
        in_channels=8,
        out_channels=4,
        down_block_types=(
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types=(
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels=(320, 640, 1280, 1280),
        addition_time_embed_dim=256,
        projection_class_embeddings_input_dim=768,
        layers_per_block=[2, 2, 2, 2],
        cross_attention_dim=(1024, 1024, 1024, 1024),
        transformer_layers_per_block=[1, 1, 1, 1],
        num_attention_heads=(5, 10, 20, 20),
        num_frames=25,
    ):
        super().__init__()
        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # time
        time_embed_dim = block_out_channels[0] * 4  # 1280

        self.time_proj = Timesteps(block_out_channels[0])
        timestep_input_dim = block_out_channels[0]  # 320

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, out_dim=time_embed_dim)
        self.add_time_proj = Timesteps(addition_time_embed_dim)
        self.add_embedding = TimestepEmbedding(
            projection_class_embeddings_input_dim, time_embed_dim, out_dim=time_embed_dim
        )

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        # # down_block_types --
        #     ['CrossAttnDownBlockSpatioTemporal', 'CrossAttnDownBlockSpatioTemporal',
        #     'CrossAttnDownBlockSpatioTemporal', 'DownBlockSpatioTemporal']
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if i < 3:  # i == 0, 1, 2
                down_block = CrossAttnDownBlockSpatioTemporal(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim[i],
                    num_attention_heads=num_attention_heads[i],
                )
            else:  # i == 3
                down_block = DownBlockSpatioTemporal(
                    num_layers=layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
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
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if i == 0:  # i == 0
                up_block = UpBlockSpatioTemporal(
                    num_layers=reversed_layers_per_block[i] + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                )
            else:  # i == 1, 2, 3
                up_block = CrossAttnUpBlockSpatioTemporal(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=reversed_layers_per_block[i] + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    add_upsample=(i != len(block_out_channels) - 1),
                    cross_attention_dim=reversed_cross_attention_dim[i],
                    num_attention_heads=reversed_num_attention_heads[i],
                )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
        # self.conv_norm_out -- GroupNorm(32, 320, eps=1e-05, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
        # self.conv_out -- Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.load_weights()

    def load_weights(self, model_path="models/unet_3d.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)
        # new_sd = {}
        # for n, p in sd.items():
        #     n = n.replace(".processor.", ".")
        #     new_sd[n] = p
        # self.load_state_dict(new_sd, strict=True)
        # torch.save(new_sd, "/tmp/unet_3d.pth")
        self.load_state_dict(sd, strict=True)

    def forward(self, sample, timestep, encoder_hidden_states, added_time_ids, pose_latents=None):
        # tensor [sample] size: [1, 16, 8, 64, 64], min: -4.78125, max: 4.757812, mean: 0.000227
        # timestep --- tensor(1.637770, device='cuda:0')
        # tensor [encoder_hidden_states] size: [1, 5, 1024], min: 0.0, max: 0.0, mean: 0.0
        # tensor [added_time_ids] size: [1, 3], min: 0.020004, max: 127.0, mean: 44.340004
        # pose_latents = None

        # 1. time
        timesteps = timestep
        if len(timesteps.shape) == 0:  # True
            # ==> pdb.set_trace()
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)
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
            sample = sample + pose_latents

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        #     ['CrossAttnDownBlockSpatioTemporal', 'CrossAttnDownBlockSpatioTemporal',
        #     'CrossAttnDownBlockSpatioTemporal', 'DownBlockSpatioTemporal']

        for i, downsample_block in enumerate(self.down_blocks):
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
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            # up_block_types ---
            #     ['UpBlockSpatioTemporal', 'CrossAttnUpBlockSpatioTemporal',
            #     'CrossAttnUpBlockSpatioTemporal', 'CrossAttnUpBlockSpatioTemporal']
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                # upsample_block -- UpBlockSpatioTemporal
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,  # diff !!!
                    image_only_indicator=image_only_indicator,
                )
            else:
                # upsample_block -- CrossAttnUpBlockSpatioTemporal
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

        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels  # embedding_dim

    def forward(self, timesteps):
        max_period = 10000

        half_dim = self.num_channels // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        # concat sine and cosine embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # flip sine and cosine embeddings
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        # zero pad
        if self.num_channels % 2 == 1:
            pdb.set_trace()
            emb = nn.functional.pad(emb, (0, 1, 0, 0))

        return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels=320,
        time_embed_dim=1280,
        out_dim=None,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, out_dim, True)

    def forward(self, sample, condition=None):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        return sample


class AlphaBlender(nn.Module):
    r"""
    A module to blend spatial and temporal features.
    """

    def __init__(self, alpha):
        super().__init__()
        self.register_parameter("mix_factor", nn.Parameter(torch.Tensor([alpha])))

    def get_alpha(self, image_only_indicator, ndims):
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

    def forward(
        self,
        x_spatial,
        x_temporal,
        image_only_indicator=None,
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
    """
    # AnimationAttention, AnimationIDAttention
    """

    def __init__(
        self,
        dim=320,
        num_attention_heads=5,
        attention_head_dim=64,
        cross_attention_dim=1024,
    ):
        super().__init__()

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-5)
        self.attn1 = AnimationAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=False,
            cross_attention_dim=None,
        )

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(dim, 1e-5, elementwise_affine=True)
        self.attn2 = AnimationIDAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=False,
        )  # AnimationIDAttention

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, 1e-5, elementwise_affine=True)
        self.ff = FeedForward(dim, dim_out=dim)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
    ):
        # Notice that normalization is always applied before
        # 1. Self-Attention
        batch_size = hidden_states.shape[0]
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states


class TransformerSpatioTemporalModel(nn.Module):
    """A Transformer model for video like data."""

    def __init__(
        self,
        num_attention_heads=5,
        attention_head_dim=64,
        in_channels=320,
        num_layers=1,
        cross_attention_dim=1024,
        num_tokens=4,
    ):
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim  # 320
        self.hidden_size = hidden_size

        # 2. Define input layers
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, hidden_size)

        # 3. Define transformers blocks
        # AnimationAttention, AnimationIDAttention
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for d in range(num_layers)
            ]
        )

        # TemporalAttention() ?
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(
                    hidden_size,
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.time_pos_embed = TimestepEmbedding(in_channels, in_channels * 4, out_dim=in_channels)
        # self.time_pos_embed
        # TimestepEmbedding(
        #   (linear_1): Linear(in_features=320, out_features=1280, bias=True)
        #   (act): SiLU()
        #   (linear_2): Linear(in_features=1280, out_features=320, bias=True)
        # )
        self.time_proj = Timesteps(in_channels)
        self.time_mixer = AlphaBlender(alpha=0.5)

        # 4. Define output layers
        self.proj_out = nn.Linear(hidden_size, in_channels)
        # self.proj_out -- Linear(in_features=320, out_features=320, bias=True)

        self.gradient_checkpointing = False
        self.num_tokens = num_tokens

    def forward(self, hidden_states, encoder_hidden_states, image_only_indicator):
        # tensor [hidden_states] size: [16, 320, 64, 64], min: -17.109375, max: 39.46875, mean: 0.010112
        # tensor [encoder_hidden_states] size: [16, 5, 1024], min: -14.492188, max: 14.453125, mean: 0.000888
        # tensor [image_only_indicator] size: [1, 16], min: 0.0, max: 0.0, mean: 0.0

        # 1. Input
        batch_frames, _, height, width = hidden_states.shape  # size: [16, 320, 64, 64]
        num_frames = image_only_indicator.shape[-1]  # size: [1, 16]
        batch_size = batch_frames // num_frames  # ==> 1

        end_pos = encoder_hidden_states.shape[1] - self.num_tokens  # ==> 1
        time_context = encoder_hidden_states[:, :end_pos, :]
        # tensor [time_context1] size: [16, 1, 1024], min: -5.863281, max: 6.507812, mean: 0.004285

        time_context_first_timestep = time_context[None, :].reshape(batch_size, num_frames, -1, time_context.shape[-1])[
            :, 0
        ]
        # tensor [time_context_first_timestep] size: [1, 1, 1024], min: -5.863281, max: 6.507812, mean: 0.004285

        time_context = time_context_first_timestep[:, None].broadcast_to(
            batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
        )
        # tensor [time_context] size: [1, 4096, 1, 1024], min: -5.863281, max: 6.507812, mean: 0.004285
        time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1])
        # tensor [time_context] size: [4096, 1, 1024], min: -5.863281, max: 6.507812, mean: 0.004285

        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_size = hidden_states.shape[1]

        # tensor [hidden_states1] size: [16, 320, 64, 64], min: -2.525391, max: 2.4375, mean: -0.013387
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, hidden_size)
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
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states_mix = hidden_states + emb

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
        # tensor [hidden_states] size: [16, 4096, 320], min: -28.609375, max: 41.75, mean: -1.081215
        hidden_states = hidden_states.reshape(batch_frames, height, width, hidden_size).permute(0, 3, 1, 2).contiguous()
        # tensor [hidden_states] size: [16, 320, 64, 64], min: -28.609375, max: 41.75, mean: -1.081215

        output = hidden_states + residual
        # tensor [output] size: [16, 320, 64, 64], min: -35.4375, max: 79.375, mean: -1.071102

        return (output,)


class FeedForward(nn.Module):
    def __init__(self, dim=320, dim_out=None, mult=4):
        super().__init__()
        hidden_size = int(dim * mult)  # 1280
        act_fn = GEGLU(dim, hidden_size)

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(0.0))
        self.net.append(nn.Linear(hidden_size, dim_out, bias=True))
        # FeedForward(
        #   (net): ModuleList(
        #     (0): GEGLU(
        #       (proj): Linear(in_features=320, out_features=2560, bias=True)
        #     )
        #     (1): Dropout(p=0.0, inplace=False)
        #     (2): Linear(in_features=1280, out_features=320, bias=True)
        #   )
        # )

    def forward(self, hidden_states):
        for m in self.net:
            hidden_states = m(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.
    """

    def __init__(self, dim_in=320, dim_out=1280):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class TemporalBasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block for video like data.
    """

    def __init__(
        self,
        dim=320,
        time_mix_inner_dim=320,
        num_attention_heads=5,
        attention_head_dim=64,
        cross_attention_dim=1024,
    ):
        super().__init__()

        assert cross_attention_dim is not None
        self.norm_in = nn.LayerNorm(dim)

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.ff_in = FeedForward(dim, dim_out=time_mix_inner_dim)

        self.norm1 = nn.LayerNorm(time_mix_inner_dim)
        self.attn1 = TemporalAttention(
            query_dim=time_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(time_mix_inner_dim)
        self.attn2 = TemporalAttention(
            query_dim=time_mix_inner_dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
        )  # is self-attn if encoder_hidden_states is none

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(time_mix_inner_dim)
        self.ff = FeedForward(time_mix_inner_dim, dim_out=time_mix_inner_dim)

    def forward(self, hidden_states, num_frames, encoder_hidden_states=None):
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
        hidden_states = ff_output + hidden_states

        hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * num_frames, seq_length, channels)

        return hidden_states


class UNetMidBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels=1280,
        temb_channels=1280,
        num_layers=1,
        transformer_layers_per_block=1,
        num_attention_heads=1,
        cross_attention_dim=1024,
    ):
        super().__init__()
        # in_channels = 1280
        # temb_channels = 1280
        # num_layers = 1
        # transformer_layers_per_block = [1]
        # num_attention_heads = 20
        # cross_attention_dim = 1024
        assert num_layers == 1
        assert cross_attention_dim == 1024

        self.has_cross_attention = True

        # support for variable transformer layers per block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        # there is always at least one resnet
        resnets = [
            SpatioTemporalResBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=1e-5,
            )
        ]
        attentions = []

        for i in range(num_layers):  # 1
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                )
            )

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, image_only_indicator=None):
        hidden_states = self.resnets[0](hidden_states, temb, image_only_indicator=image_only_indicator)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
            )[0]
            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )

        return hidden_states


class DownBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels=1280,
        out_channels=1280,
        temb_channels=1280,
        num_layers=2,
    ):
        super().__init__()
        assert num_layers == 2

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                )
            )
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, image_only_indicator=None):
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, image_only_indicator=image_only_indicator)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


# --- down_block_types
class CrossAttnDownBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels=320,
        out_channels=320,
        temb_channels=1280,
        num_layers=2,
        transformer_layers_per_block=1,
        num_attention_heads=1,
        cross_attention_dim=1024,
    ):
        super().__init__()

        # assert num_attention_heads == 5
        assert cross_attention_dim == 1024
        assert num_layers == 2

        resnets = []
        attentions = []

        self.has_cross_attention = True
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=1e-6,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = nn.ModuleList(
            [
                Downsample2D(
                    out_channels,
                    out_channels=out_channels,
                    # padding=1,
                    name="op",
                )
            ]
        )

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, image_only_indicator=None):
        output_states = ()

        blocks = list(zip(self.resnets, self.attentions))
        for resnet, attn in blocks:
            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )
            # attn -- TransformerSpatioTemporalModel
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
            )[0]

            output_states = output_states + (hidden_states,)

        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)
        output_states = output_states + (hidden_states,)

        return hidden_states, output_states


# up_block_types
class UpBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels=1280,
        prev_output_channel=1280,
        out_channels=1280,
        temb_channels=1280,
        num_layers=1,
        resnet_eps=1e-6,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels=out_channels)])

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        image_only_indicator=None,
    ):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )

        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)

        return hidden_states


# up_block_types
class CrossAttnUpBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        prev_output_channel,
        temb_channels,
        num_layers=1,
        transformer_layers_per_block=1,
        resnet_eps=1e-6,
        num_attention_heads=1,
        cross_attention_dim=1024,
        add_upsample=True,
    ):
        super().__init__()

        assert num_layers == 3

        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):  # 3 ?
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        # assert add_downsample == True
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels=out_channels)])
        else:
            self.upsamplers = None

        # pdb.set_trace()

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        image_only_indicator=None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
            )[0]

        if self.upsamplers is not None:  # True
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        else:
            # ==> pdb.set_trace()
            pass

        # tensor [hidden_states] size: [16, 1280, 32, 32], min: -185.625, max: 165.875, mean: 0.257755

        return hidden_states


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.
    out_channels,
    out_channels=out_channels,
    name="op",
    """

    def __init__(
        self,
        channels,
        out_channels=None,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels  # !!!
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        return hidden_states


class TemporalConvLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=None,
        norm_num_groups=32,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, in_dim),
            nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, hidden_states, num_frames=1):
        hidden_states = (
            hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        )

        identity = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)

        hidden_states = identity + hidden_states

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )
        return hidden_states


class SpatioTemporalResBlock(nn.Module):
    """
    in_channels=in_channels,
    out_channels=in_channels,
    temb_channels=temb_channels,
    eps=1e-5,
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        temb_channels=512,
        eps=1e-6,
        merge_factor=0.5,
    ):
        super().__init__()

        self.spatial_res_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=eps,
        )

        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels,  # if out_channels is not None else in_channels,
            out_channels=out_channels,  # if out_channels is not None else in_channels,
            temb_channels=temb_channels,
            eps=eps,
        )

        self.time_mixer = AlphaBlender(alpha=merge_factor)

    def forward(
        self,
        hidden_states,
        temb=None,
        image_only_indicator=None,
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


class ResnetBlock2D(nn.Module):
    r"""
    in_channels=in_channels,
    out_channels=out_channels,
    temb_channels=temb_channels,
    eps=eps,
    """

    def __init__(
        self,
        in_channels=320,
        out_channels=320,
        temb_channels=512,
        eps=1e-6,
        groups=32,
    ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()

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
        # pdb.set_trace()

    def forward(self, input_tensor, temb, *args, **kwargs):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None:
            hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states  # / self.output_scale_factor

        return output_tensor


class TemporalResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels=320,
        out_channels=None,
        temb_channels=None,
        eps=1e-6,
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

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True)
        # xxxx_debug
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        # pdb.set_trace()

    def forward(self, input_tensor, temb):
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


class Upsample2D(nn.Module):
    def __init__(
        self,
        channels=1280,
        out_channels=1280,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, hidden_states, output_size=None):
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

        hidden_states = self.conv(hidden_states)

        return hidden_states


class TemporalAttention(nn.Module):
    r"""
    Processor for implementing memory efficient attention using xFormers.
    """

    def __init__(
        self,
        query_dim=320,
        cross_attention_dim=None,  # 1024 or None
        heads=5,
        dim_head=64,
        bias=False,
        # processor = None,
    ):
        super().__init__()
        self.head_size = heads  # !!!

        self.hidden_size = dim_head * heads  # hidden_size ???
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.residual_connection = False
        self.out_dim = query_dim
        self.scale = dim_head**-0.5

        # cross_attention_dim == 1024 or None
        self.to_q = nn.Linear(query_dim, self.hidden_size, bias=False)
        self.to_k = nn.Linear(self.cross_attention_dim, self.hidden_size, bias=False)
        self.to_v = nn.Linear(self.cross_attention_dim, self.hidden_size, bias=False)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.hidden_size, self.out_dim, bias=True))
        self.to_out.append(nn.Dropout(0.0))

    def forward(self, hidden_states, encoder_hidden_states=None):
        # tensor [hidden_states] size: [4096, 16, 320], min: -3.316406, max: 6.085938, mean: 0.003951
        # tensor [encoder_hidden_states] size: [4096, 1, 1024], min: -5.863281, max: 6.507812, mean: 0.004285

        residual = hidden_states

        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = self.head_to_batch_dim(query).contiguous()
        key = self.head_to_batch_dim(key).contiguous()
        value = self.head_to_batch_dim(value).contiguous()

        if is_xformers_available():  # True
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=None, op=None, scale=self.scale
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        # tensor [hidden_states] size: [4096, 16, 320], min: -0.095337, max: 0.110901, mean: -0.002453
        return hidden_states

    def batch_to_head_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // self.head_size, self.head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // self.head_size, seq_len, dim * self.head_size)
        return tensor

    def head_to_batch_dim(self, tensor, out_dim=3):
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.head_size, dim // self.head_size)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size * self.head_size, seq_len, dim // self.head_size)

        return tensor


class AnimationAttention(nn.Module):
    def __init__(
        self,
        query_dim=320,
        cross_attention_dim=None,  # 1024 or None
        heads=5,
        dim_head=64,
        bias=False,
        # processor = None,
    ):
        super().__init__()
        self.head_size = heads  # !!!

        self.hidden_size = dim_head * heads  # hidden_size ???
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.residual_connection = False
        self.out_dim = query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        self.group_norm = None
        self.spatial_norm = None

        # cross_attention_dim == 1024 or None
        self.norm_cross = None
        self.to_q = nn.Linear(query_dim, self.hidden_size, bias=False)
        self.to_k = nn.Linear(self.cross_attention_dim, self.hidden_size, bias=False)
        self.to_v = nn.Linear(self.cross_attention_dim, self.hidden_size, bias=False)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.hidden_size, self.out_dim, bias=True))
        self.to_out.append(nn.Dropout(0.0))

    def forward(self, hidden_states, encoder_hidden_states=None):
        # tensor [hidden_states] size: [16, 4096, 320], min: -10.609375, max: 32.09375, mean: 0.12576
        # [encoder_hidden_states] type: <class 'NoneType'>
        # [attention_mask] type: <class 'NoneType'>

        # attention_mask = None
        # temb = None
        # (Pdb) attn
        # Attention(
        #   (to_q): Linear(in_features=320, out_features=320, bias=False)
        #   (to_k): Linear(in_features=320, out_features=320, bias=False)
        #   (to_v): Linear(in_features=320, out_features=320, bias=False)
        #   (to_out): ModuleList(
        #     (0): Linear(in_features=320, out_features=320, bias=True)
        #     (1): Dropout(p=0.0, inplace=False)
        #   )
        #   (processor): AnimationAttention()
        # )
        # tensor [hidden_states] size: [16, 4096, 320], min: -0.878906, max: 1.689453, mean: 0.00032
        assert hidden_states is not None
        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            # ==> pdb.set_trace()
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = self.head_to_batch_dim(query).contiguous()
        key = self.head_to_batch_dim(key).contiguous()
        value = self.head_to_batch_dim(value).contiguous()

        if is_xformers_available():
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = self.get_attention_scores(query, key, None)
            hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)

        # linear proj
        # hidden_states = self.to_out[0](hidden_states) + self.lora_scale * self.to_out_lora(hidden_states)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        # tensor [hidden_states] size: [16, 4096, 320], min: -4.476562, max: 0.760742, mean: -0.001343

        return hidden_states

    def batch_to_head_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // self.head_size, self.head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // self.head_size, seq_len, dim * self.head_size)
        return tensor

    def head_to_batch_dim(self, tensor, out_dim=3):
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.head_size, dim // self.head_size)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size * self.head_size, seq_len, dim // self.head_size)

        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        # xxxx_debug
        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs


class AnimationIDAttention(nn.Module):
    def __init__(
        self,
        query_dim=320,
        cross_attention_dim=None,  # 1024 or None
        heads=5,
        dim_head=64,
        bias=False,
        # processor = None,
    ):
        super().__init__()
        self.head_size = heads  # !!!

        self.hidden_size = dim_head * heads  # hidden_size ???
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.residual_connection = False
        self.out_dim = query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        self.group_norm = None
        self.spatial_norm = None

        # cross_attention_dim == 1024 or None
        self.norm_cross = None
        self.to_q = nn.Linear(query_dim, self.hidden_size, bias=False)
        self.to_k = nn.Linear(self.cross_attention_dim, self.hidden_size, bias=False)
        self.to_v = nn.Linear(self.cross_attention_dim, self.hidden_size, bias=False)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.hidden_size, self.out_dim, bias=True))
        self.to_out.append(nn.Dropout(0.0))

        # self.hidden_size = hidden_size

        assert cross_attention_dim == 1024

        # self.scale = scale
        self.id_to_k = nn.Linear(cross_attention_dim, self.hidden_size, bias=False)
        self.id_to_v = nn.Linear(cross_attention_dim, self.hidden_size, bias=False)

        self.num_tokens = 4  # num_tokens
        # self = AnimationIDAttention(
        #   (id_to_k): Linear(in_features=1024, out_features=320, bias=False)
        #   (id_to_v): Linear(in_features=1024, out_features=320, bias=False)
        # )
        # assert self.scale == 1.0

    def forward(self, hidden_states, encoder_hidden_states=None):
        # tensor [hidden_states] size: [16, 4096, 320], min: -1.165039, max: 1.260742, mean: 0.035343
        # tensor [encoder_hidden_states] size: [16, 5, 1024], min: -14.492188, max: 14.453125, mean: 0.000888

        # (Pdb) attn
        # Attention(
        #   (to_q): Linear(in_features=320, out_features=320, bias=False)
        #   (to_k): Linear(in_features=1024, out_features=320, bias=False)
        #   (to_v): Linear(in_features=1024, out_features=320, bias=False)
        #   (to_out): ModuleList(
        #     (0): Linear(in_features=320, out_features=320, bias=True)
        #     (1): Dropout(p=0.0, inplace=False)
        #   )
        #   (processor): AnimationIDAttention(
        #     (id_to_k): Linear(in_features=1024, out_features=320, bias=False)
        #     (id_to_v): Linear(in_features=1024, out_features=320, bias=False)
        #   )
        # )
        assert hidden_states is not None
        assert encoder_hidden_states is not None

        residual = hidden_states

        query = self.to_q(hidden_states)
        encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)

        end_pos = encoder_hidden_states.shape[1] - self.num_tokens
        encoder_hidden_states, ip_hidden_states = (
            encoder_hidden_states[:, :end_pos, :],
            encoder_hidden_states[:, end_pos:, :],
        )

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        hidden_size = key.shape[-1]
        head_dim = hidden_size // self.heads

        query = self.head_to_batch_dim(query).contiguous()
        key = self.head_to_batch_dim(key).contiguous()
        value = self.head_to_batch_dim(value).contiguous()

        key = key.to(query.dtype)
        value = value.to(query.dtype)

        if is_xformers_available():
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
            hidden_states = hidden_states.to(query.dtype)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.batch_to_head_dim(hidden_states)

        ip_key = self.id_to_k(ip_hidden_states)
        ip_value = self.id_to_v(ip_hidden_states)

        ip_key = self.head_to_batch_dim(ip_key).contiguous()
        ip_value = self.head_to_batch_dim(ip_value).contiguous()
        ip_key = ip_key.to(query.dtype)
        ip_value = ip_value.to(query.dtype)

        if is_xformers_available():
            ip_hidden_states = xformers.ops.memory_efficient_attention(query, ip_key, ip_value, attn_bias=None)
            ip_hidden_states = ip_hidden_states.to(query.dtype)
        else:
            ip_hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            ip_hidden_states = ip_hidden_states.to(query.dtype)

        ip_hidden_states = self.batch_to_head_dim(ip_hidden_states)
        mean_latents, std_latents = torch.mean(hidden_states, dim=(1, 2), keepdim=True), torch.std(
            hidden_states, dim=(1, 2), keepdim=True
        )
        mean_ip, std_ip = torch.mean(ip_hidden_states, dim=(1, 2), keepdim=True), torch.std(
            ip_hidden_states, dim=(1, 2), keepdim=True
        )
        ip_hidden_states = (ip_hidden_states - mean_ip) * (std_latents / (std_ip + 1e-5)) + mean_latents

        hidden_states = hidden_states + ip_hidden_states

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        # tensor [hidden_states] size: [16, 4096, 320], min: -0.642578, max: 0.390625, mean: -0.000428

        return hidden_states

    def batch_to_head_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // self.head_size, self.head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // self.head_size, seq_len, dim * self.head_size)
        return tensor

    def head_to_batch_dim(self, tensor, out_dim=3):
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.head_size, dim // self.head_size)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size * self.head_size, seq_len, dim // self.head_size)

        return tensor


if __name__ == "__main__":
    device = torch.device("cuda")
    unet = UNetSpatioTemporalConditionModel()
    unet.eval
    unet = unet.to(device)
    # print(unet)

    # tensor [sample] size: [1, 16, 8, 64, 64], min: -4.78125, max: 4.757812, mean: 0.000227
    # timestep --- tensor(1.637770, device='cuda:0')
    # tensor [encoder_hidden_states] size: [1, 5, 1024], min: 0.0, max: 0.0, mean: 0.0
    # tensor [added_time_ids] size: [1, 3], min: 0.020004, max: 127.0, mean: 44.340004
    # pose_latents = None
    # image_only_indicator = False
    sample = torch.randn(1, 16, 8, 64, 64).to(device)
    timesteps = torch.randn(1).to(device)
    encoder_hidden_states = torch.randn(1, 5, 1024).to(device)
    added_time_ids = torch.randn(1, 3).to(device)
    # pose_latents = None
    # image_only_indicator = False

    with torch.no_grad():
        # unet_output = unet(sample, timesteps, encoder_hidden_states, added_time_ids, pose_latents, image_only_indicator)
        unet_output = unet(sample, timesteps, encoder_hidden_states, added_time_ids)

    todos.debug.output_var("unet_output", unet_output)
