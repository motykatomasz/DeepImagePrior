import torch
import torch.nn as nn
from models.unet_modules import Down, Up, Skip
from models.utils import get_padding_by_kernel

class UNet(nn.Module):
    def __init__(self, channels, config):
        super(UNet, self).__init__()
        self.channels_down = config["channels_down"]
        self.channels_up = config["channels_up"]
        self.channels_skip = config["channels_skip"]

        self.kernels_down = config["kernels_down"]
        self.kernels_up = config["kernels_up"]
        self.kernels_skip = config["kernels_skip"]

        self.upsampling_method = config["upsampling_method"]

        self.down = nn.ModuleList()
        for i in range(len(self.channels_down)):
            self.down.append(
                Down(
                    self.channels_down[i - 1] if i > 0 else channels,
                    self.channels_down[i],
                    self.kernels_down[i]
                )
            )

        self.skip = nn.ModuleList()
        for i in range(len(self.channels_skip)):
            if self.channels_skip[i] > 0:
                self.skip.append(
                    Skip(
                        self.channels_down[i],
                        self.channels_skip[i],
                        self.kernels_skip[i]
                    )
                )

        self.up = nn.ModuleList()
        for i in range(len(self.channels_up)):
            self.up.append(
                Up(
                    self.channels_up[i] + self.channels_skip[i],
                    self.channels_up[i - 1] if i > 0 else 3,
                    self.kernels_up[i],
                    self.upsampling_method
                )
            )

        self.debug_convolution = nn.Sequential(
            nn.Conv2d(3, 3, self.kernels_up[0], padding=get_padding_by_kernel(self.kernels_up[0])),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = x
        x_downsampled = []
        for i in range(len(self.channels_down)):
                out = self.down[i](out)
                x_downsampled.append(out)

        for i in reversed(range(len(self.channels_up))):
            if self.channels_skip[i] > 0:
                out = self.up[i](torch.cat([out, self.skip[i](x_downsampled[i])], dim=1))
            else:
                out = self.up[i](out)

        return self.debug_convolution(out)
