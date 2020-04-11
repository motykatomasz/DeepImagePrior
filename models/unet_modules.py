import torch.nn as nn
from models.utils import get_padding_by_kernel


class Down(nn.Module):
    '''Class representing 1 segment of downsampling process. It consists of 2 conv layers with LeakyReLu activations
    followed by maxed pooling layer, which leads to the next such segement. Implemented as a sequential
    container added to UNet. (Things like dropout to add later)'''

    def __init__(self, channels_in, channels_out, kernel_size, activation=nn.LeakyReLU()):
        super(Down, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, padding=get_padding_by_kernel(kernel_size)),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(channels_out),
            activation,

            nn.Conv2d(channels_out, channels_out, kernel_size, padding=get_padding_by_kernel(kernel_size)),
            nn.BatchNorm2d(channels_out),
            activation,
        )

    def forward(self, x):
        return self.downsample(x)


class Up(nn.Module):
    '''Class representing 1 segment of upsampling process. It consists of 2 conv layers with LeakyReLu activations
    followed by upsampling layer, which leads to the next such segement. Implemented as a sequential
    container added to UNet. (Things like dropout to add later)'''

    def __init__(self, channels_in, channels_out, kernel_size, upsampling_method, activation=nn.LeakyReLU()):
        super(Up, self).__init__()

        self.upsample = nn.Sequential(
            nn.BatchNorm2d(channels_in),

            # Input channels are added to account for the concatenation with the output from skip connection
            nn.Conv2d(channels_in, channels_out, kernel_size, padding=get_padding_by_kernel(kernel_size)),
            nn.BatchNorm2d(channels_out),
            activation,

            nn.Conv2d(channels_out, channels_out, 1),
            nn.BatchNorm2d(channels_out),
            activation,

            nn.Upsample(scale_factor=2, mode=upsampling_method)
        )

    def forward(self, x):
        return self.upsample(x)

class Skip(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, activation=nn.LeakyReLU()):
        super(Skip, self).__init__()

        self.skipsample = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, padding=get_padding_by_kernel(kernel_size)),
            nn.BatchNorm2d(channels_out),
            activation,
        )

    def forward(self, x):
        return self.skipsample(x)


class OutConv(nn.Module):
    def __init__(self, channels_in, channels_out=3):
        super(OutConv, self).__init__()

        self.outconv = nn.Sequential(
            # UNet uses single value kernel
            nn.Conv2d(channels_in, channels_out, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.outconv(x)