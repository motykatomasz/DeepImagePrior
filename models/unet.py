import torch
import torch.nn as nn
from models.unet_modules import Down, Up, Skip, OutConv, Connect


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
            else:
                self.skip.append(None)

        self.connect_layer = Connect(
            self.channels_down[-1],
            self.channels_up[-1],
            self.kernels_down[-1],
            self.upsampling_method
        )

        self.up = nn.ModuleList()
        for i in range(len(self.channels_up)):
            self.up.append(
                Up(
                    self.channels_up[i] + self.channels_skip[i],
                    self.channels_up[i - 1] if i > 0 else self.channels_up[i],
                    self.kernels_up[i],
                    self.upsampling_method
                )
            )

        self.out_conv = OutConv(self.channels_up[0])

    def forward(self, x):
        out = x
        x_downsampled = []
        for i in range(len(self.channels_down)):
            out = self.down[i](out)
            x_downsampled.append(out)

        out = self.connect_layer(out)

        for i in reversed(range(len(self.channels_up))):
            if self.channels_skip[i] > 0:
                out = self.up[i](torch.cat([out, self.skip[i](x_downsampled[i])], dim=1))
            else:
                out = self.up[i](out)

        return self.out_conv(out)
