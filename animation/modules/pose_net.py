import einops
import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
import pdb
import todos

# xxxx_debug
class PoseNet(ModelMixin):
# class PoseNet(nn.Module):
    def __init__(self, noise_latent_channels=320):
        super().__init__()
        # multiple convolution layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )

        # Final projection layer
        self.final_proj = nn.Conv2d(in_channels=128, out_channels=noise_latent_channels, kernel_size=1)
        self.scale = nn.Parameter(torch.ones(1) * 2)

    def forward(self, x):
        # tensor [x] size: [16, 3, 512, 512], min: -1.0, max: 1.0, mean: -0.994078

        x = self.conv_layers(x)
        x = self.final_proj(x)
        # tensor [x] size: [16, 320, 64, 64], min: -0.747559, max: 0.661133, mean: 0.000168

        return x * self.scale


