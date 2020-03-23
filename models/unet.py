import torch.nn as nn
from models.unet_modules import Down, Up


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.channels_down = config["channels_down"]
        self.channels_up = config["channels_up"]
        self.channels_skip = config["channels_skip"]

        self.kernels_down = config["kernels_down"]
        self.kernels_up = config["kernels_up"]
        self.kernels_skip = config["kernels_skip"]

        self.upsampling_method = config["upsampling_method"]

        self.down = nn.ModuleList()
        for i in range(len(self.channels_down)-1):
            self.down.append(Down(self.channels_down[i], self.channels_down[i+1], self.kernels_down[i]))

        self.up = nn.ModuleList()
        for i in range(len(self.channels_up)-1):
            self.up.append(Up(self.channels_up[i], self.channels_up[i+1], self.channels_skip[i], self.kernels_up[i],
                              self.kernels_skip[i], self.upsampling_method))

    def forward(self, x):
        x_downsampled = [x]
        for i in range(len(self.channels_down)-1):
            x_downsampled.append(self.down[i](x_downsampled[i]))

        out = self.up[0](x_downsampled[-1], x_downsampled[-1])
        for i in range(1, len(self.channels_up)-1):
            out = self.up[i](out, x_downsampled[-(i+1)])

        return out
