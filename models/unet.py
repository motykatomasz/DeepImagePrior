import torch.nn as nn
from models.unet_modules import Down, Up


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Dummy values right now
        self.down1 = Down(3, 16, 0.01)
        self.down2 = Down(16, 32, 0.01)
        self.down3 = Down(32, 64, 0.01)
        self.down4 = Down(64, 128, 0.01)
        self.down5 = Down(128, 128, 0.01)
        self.down6 = Down(128, 128, 0.01)

        self.up1 = Up(128, 128, 5, 0.01)
        self.up2 = Up(128, 128, 5, 0.01)
        self.up3 = Up(128, 64, 5, 0.01)
        self.up4 = Up(64, 32, 5, 0.01)
        self.up5 = Up(32, 16, 5, 0.01)
        self.up6 = Up(16, 3, 5, 0.01)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)

        x = self.up1(x6)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        out = self.up6(x)

        return out


if __name__ == "__main__":
    net = UNet()