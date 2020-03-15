import torch.nn as nn
from models.unet_modules import Down
from models.configs import sample_config


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Dummy values right now
        self.down1 = Down(10, 10, 0.01)
        self.down2 = Down(10, 10, 0.01)
        self.down3 = Down(10, 10, 0.01)


    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        return 0





if __name__ == "__main__":
    net = UNet(sample_config)