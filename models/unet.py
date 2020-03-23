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

        self.down1 = Down(32, self.channels_down[0], self.kernels_down[0])
        self.down2 = Down(self.channels_down[0], self.channels_down[1], self.kernels_down[1])
        self.down3 = Down(self.channels_down[1], self.channels_down[2], self.kernels_down[2])
        self.down4 = Down(self.channels_down[2], self.channels_down[3], self.kernels_down[3])
        self.down5 = Down(self.channels_down[3], self.channels_down[4], self.kernels_down[4])
        self.down6 = Down(self.channels_down[4], self.channels_down[5], self.kernels_down[5])

        self.up1 = Up(self.channels_up[0], self.channels_up[1], self.channels_skip[0], self.kernels_up[0],
                      self.kernels_skip[0], self.upsampling_method)
        self.up2 = Up(self.channels_up[1], self.channels_up[2], self.channels_skip[1], self.kernels_up[1],
                      self.kernels_skip[1], self.upsampling_method)
        self.up3 = Up(self.channels_up[2], self.channels_up[3], self.channels_skip[2], self.kernels_up[2],
                      self.kernels_skip[2], self.upsampling_method)
        self.up4 = Up(self.channels_up[3], self.channels_up[4], self.channels_skip[3], self.kernels_up[3],
                      self.kernels_skip[3], self.upsampling_method)
        self.up5 = Up(self.channels_up[4], self.channels_up[5], self.channels_skip[4], self.kernels_up[4],
                      self.kernels_skip[4], self.upsampling_method)
        self.up6 = Up(self.channels_up[5], 3, self.channels_skip[5], self.kernels_up[5], self.kernels_skip[5],
                      self.upsampling_method)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)

        x = self.up1(x6, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        out = self.up6(x, x1)

        return out
