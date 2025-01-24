import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import todos

# --------------------
class AutoencoderKLTemporalDecoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        latent_channels=4,
        # -----------------------------------
        sample_size=768,
        scaling_factor=0.18215,
    ):
        super().__init__()
        self.MAX_H = 512  # Fixed, DO NOT change !!!
        self.MAX_W = 512  # Fixed, DO NOT change !!!
        self.MAX_TIMES = 1

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
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

        self.tile_latent_min_size = 96
        self.tile_overlap_factor = 0.25

        self.load_weights()

    def load_weights(self, model_path="models/vae_3d.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint), strict=True)

    def encode(self, x):
        # tensor [x] size: [1, 3, 512, 512], min: -1.085592, max: 1.066735, mean: -0.045391
        h = self.encoder(x)
        moments = self.quant_conv(h)
        # tensor [moments] size: [1, 8, 64, 64], min: -56.005188, max: 35.71368, mean: -9.935047

        return moments
        # posterior = DiagonalGaussianDistribution(moments)
        # return posterior

    def decode(self, z, num_frames):
        # return_dict = True
        assert num_frames == 4
        # tensor [decode z] size: [4, 4, 64, 64], min: -35.8125, max: 41.1875, mean: -0.8983
        batch_size = z.shape[0] // num_frames

        # xxxx_debug !!!
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=z.dtype, device=z.device)
        decoded = self.decoder(z, num_frames=num_frames, image_only_indicator=image_only_indicator)
        # tensor [decoded] size: [4, 3, 512, 512], min: -1.092773, max: 1.15332, mean: -0.015342

        return decoded

    def forward(
        self,
        sample,
        sample_posterior=False,
        generator=None,
        num_frames=1,
    ):
        r"""useless ..."""
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        dec = self.decode(z, num_frames=num_frames).sample

        return dec


def randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
    """
    sample = randn_tensor(
        self.mean.shape,
        generator=generator,
        device=self.parameters.device,
        dtype=self.parameters.dtype,
    )
    """
    layout = layout or torch.strided
    device = device or torch.device("cpu")
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype, layout=layout).to(device)

    return latents


class Attention(nn.Module):
    def __init__(
        self,
        query_dim,
        heads=8,
        dim_head=64,
    ):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.cross_attention_dim = query_dim
        self.out_dim = query_dim
        self.scale = dim_head**-0.5

        self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=32, eps=1e-6, affine=True)

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=True)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=True))
        self.to_out.append(nn.Dropout(0.0))

    def forward(self, hidden_states):
        # tensor [hidden_states] size: [1, 512, 64, 64], min: -656.834106, max: 557.63092, mean: 0.08263

        residual = hidden_states
        input_ndim = hidden_states.ndim

        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        if self.group_norm is not None:
            # ==> pdb.set_trace()
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        # hidden_states = self.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        hidden_states = hidden_states + residual
        # tensor [hidden_states] size: [1, 512, 64, 64], min: -668.319702, max: 548.596924, mean: 0.415326

        return hidden_states


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters):
        # tensor [parameters] size: [1, 8, 64, 64], min: -56.005188, max: 35.71368, mean: -9.935047
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # ==> self.mean.size() -- [1, 4, 64, 64]
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, generator=None):
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape, generator=generator, device=self.parameters.device, dtype=self.parameters.dtype
        )
        x = self.mean + self.std * sample
        # todos.debug.output_var("sample", x)
        return x

    def mode(self):
        return self.mean


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels=512,
        attention_head_dim=512,
        num_layers=1,
    ):
        super().__init__()
        resnets = [ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, eps=1e-6, groups=32)]
        attentions = []

        assert num_layers == 1
        for _ in range(num_layers):
            attentions.append(
                Attention(
                    in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                )
            )
            resnets.append(
                ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, eps=1e-6, groups=32)
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states):
        # tensor [hidden_states1] size: [1, 512, 64, 64], min: -524.558533, max: 458.682739, mean: 1.415484

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
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states)
        # tensor [hidden_states2] size: [1, 512, 64, 64], min: -631.920898, max: 525.334229, mean: 0.462752

        return hidden_states


class TemporalDecoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, block_out_channels=(128, 256, 512, 512), layers_per_block=2):
        super().__init__()
        assert layers_per_block == 2
        first_in_channel = 128 # block_out_channels[0]
        last_out_channel = 512 # block_out_channels[-1]
        self.conv_in = nn.Conv2d(in_channels, last_out_channel, kernel_size=3, stride=1, padding=1)
        self.mid_block = MidBlockTemporalDecoder(num_layers=layers_per_block, dim=last_out_channel)

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):  #  block_out_channels -- [128, 256, 512, 512]
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

        self.conv_norm_out = nn.GroupNorm(num_channels=first_in_channel, num_groups=32, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            in_channels=first_in_channel, out_channels=out_channels, kernel_size=3, padding=1
        )

        out_kernel_size = (3, 1, 1)
        padding = [int(k // 2) for k in out_kernel_size]  # [1, 0, 0]
        self.time_conv_out = nn.Conv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=out_kernel_size, padding=padding
        )

    def forward(self, sample, image_only_indicator, num_frames=1):
        # tensor [sample] size: [4, 4, 64, 64], min: -3.986736, max: 4.073826, mean: 0.00144
        # tensor [image_only_indicator] size: [1, 4], min: 0.0, max: 0.0, mean: 0.0
        # [num_frames] value: '4'

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
        # tensor [sample] size: [1, 3, 4, 512, 512], min: -5.296875, max: 3.916016, mean: -0.315182
        sample = self.time_conv_out(sample)
        # tensor [sample] size: [1, 3, 4, 512, 512], min: -1.197266, max: 1.191406, mean: -0.004497
        sample = sample.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        # tensor [sample] size: [4, 3, 512, 512], min: -0.477861, max: 0.189522, mean: -0.111536

        return sample


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
    ):
        super().__init__()
        output_channel = 128 # block_out_channels[0]
        last_out_channel = 512 # block_out_channels[-1];

        self.conv_in = nn.Conv2d(in_channels, output_channel, kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList([])
        for i, down_block_type in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = DownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block,
                add_downsample=not is_final_block,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(in_channels=last_out_channel, attention_head_dim=last_out_channel)

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=last_out_channel, num_groups=32, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(last_out_channel, 2 * out_channels, 3, padding=1)

    def forward(self, sample):
        # tensor [sample] size: [1, 3, 512, 512], min: -4.833205, max: 4.741517, mean: -0.000397
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

        # tensor [sample] size: [1, 8, 64, 64], min: -12.912671, max: 20.121521, mean: 2.41029
        return sample


class MidBlockTemporalDecoder(nn.Module):
    def __init__(self, dim=512, num_layers=2):
        super().__init__()
        resnets = []
        attentions = []
        assert num_layers == 2
        for i in range(num_layers):  # 2
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=dim, out_channels=dim, eps=1e-6, merge_factor=0.0
                )
            )
        attentions.append(
            Attention(
                query_dim=dim,
                heads=dim // dim, # 1
                dim_head=dim,
            )
        )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, image_only_indicator):
        # tensor [hidden_states] size: [4, 512, 64, 64], min: -2.799238, max: 2.383698, mean: 0.012568
        # tensor [image_only_indicator] size: [1, 4], min: 0.0, max: 0.0, mean: 0.0
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
        # tensor [hidden_states] size: [4, 512, 64, 64], min: -7.462655, max: 6.545697, mean: 0.0344

        return hidden_states


class UpBlockTemporalDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=3,
        add_upsample=True,  # True or False
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=input_channels, out_channels=out_channels, eps=1e-6, merge_factor=0.0
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, image_only_indicator):
        # tensor [hidden_states] size: [4, 256, 512, 512], min: -1035.322632, max: 813.782288, mean: -2.003695
        # tensor [image_only_indicator] size: [1, 4], min: 0.0, max: 0.0, mean: 0.0
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, image_only_indicator=image_only_indicator)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        # tensor [hidden_states] size: [4, 128, 512, 512], min: -2852.224121, max: 2672.371338, mean: -11.729029
        return hidden_states


# --------------------------------------
class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=1,
        add_downsample=True,  # True | False
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, eps=1e-6, groups=32)
            )

        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels=out_channels, padding=0)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class Downsample2D(nn.Module):
    def __init__(self, channels, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(self.channels, out_channels, kernel_size=3, stride=2, padding=padding, bias=True)

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels

        pad = (0, 1, 0, 1)
        hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        # assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


# !!! ---------------------------------------
class Upsample2D(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(self.channels, out_channels, kernel_size=3, padding=1, bias=True)

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


# !!! ---------------------------------
class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32, eps=1e-6):
        super().__init__()

        self.in_channels = in_channels
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        self.use_in_shortcut = self.in_channels != out_channels

        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.conv_shortcut = None

    def forward(self, input_tensor):
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


# !!! ---------------------------------
class TemporalResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, eps=1e-6):
        super().__init__()

        self.in_channels = in_channels

        kernel_size = (3, 1, 1)
        padding = [k // 2 for k in kernel_size]
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

        # xxxx_debug
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

        self.nonlinearity = nn.SiLU()  # get_activation("silu")

        self.use_in_shortcut = self.in_channels != out_channels
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

    def forward(self, input_tensor):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        # tensor [hidden_states] size: [1, 512, 4, 64, 64], min: -0.278564, max: 33.3125, mean: 0.150598
        hidden_states = self.conv1(hidden_states)
        # tensor [hidden_states] size: [1, 512, 4, 64, 64], min: -193.625, max: 17.5, mean: 0.080203

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        # tensor [hidden_states] size: [1, 512, 4, 64, 64], min: -0.278564, max: 17.9375, mean: 0.021441
        hidden_states = self.conv2(hidden_states)
        # tensor [hidden_states] size: [1, 512, 4, 64, 64], min: -16.4375, max: 15.179688, mean: -0.033997

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


# VideoResBlock
class SpatioTemporalResBlock(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, eps=1e-6, merge_factor=0.0):
        super().__init__()
        self.spatial_res_block = ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, eps=eps)
        self.temporal_res_block = TemporalResnetBlock(in_channels=out_channels, out_channels=out_channels)
        self.time_mixer = AlphaBlender(alpha=merge_factor)

    def forward(self, hidden_states, image_only_indicator):
        # tensor [hidden_states] size: [4, 128, 512, 512], min: -2456.013672, max: 2166.745117, mean: -12.363848
        # tensor [image_only_indicator] size: [1, 4], min: 0.0, max: 0.0, mean: 0.0

        num_frames = image_only_indicator.shape[-1]
        hidden_states = self.spatial_res_block(hidden_states)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )
        # tensor [hidden_states_mix] size: [1, 128, 4, 512, 512], min: -2725.466553, max: 2433.611328, mean: -13.435627

        # tensor [hidden_states] size: [4, 128, 512, 512], min: -2725.466553, max: 2433.611328, mean: -13.435627
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )
        # tensor [hidden_states] size: [1, 128, 4, 512, 512], min: -2725.466553, max: 2433.611328, mean: -13.435627

        hidden_states = self.temporal_res_block(hidden_states)
        hidden_states = self.time_mixer(x_spatial=hidden_states_mix, x_temporal=hidden_states)

        # tensor [hidden_states] size: [1, 128, 4, 512, 512], min: -2725.077637, max: 2434.273193, mean: -13.265444
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        # tensor [hidden_states] size: [4, 128, 512, 512], min: -2725.077637, max: 2434.273193, mean: -13.265444

        return hidden_states


class AlphaBlender(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.register_parameter("mix_factor", nn.Parameter(torch.Tensor([alpha])))

    def forward(self, x_spatial, x_temporal):
        alpha = torch.sigmoid(self.mix_factor)
        alpha = alpha.to(x_spatial.dtype)
        alpha = 1.0 - alpha

        x = alpha * x_spatial + (1.0 - alpha) * x_temporal
        return x


if __name__ == "__main__":
    device = torch.device("cuda")
    vae = AutoencoderKLTemporalDecoder()
    vae.eval()
    vae.to(device)
    # print(vae)

    # tensor [x] size: [1, 3, 512, 512], min: -1.085592, max: 1.066735, mean: -0.045391
    x = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        moments = vae.encode(x)

    todos.debug.output_var("moments", moments)

    # tensor [decode z] size: [4, 4, 64, 64], min: -35.8125, max: 41.1875, mean: -0.8983
    z = torch.randn(4, 4, 64, 64).to(device)
    with torch.no_grad():
        x = vae.decode(z, 4)

    todos.debug.output_var("x", x)
