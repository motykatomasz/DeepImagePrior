import torch
import torch.nn as nn
from models.utils import get_padding_by_kernel


class Down(nn.Module):
    '''Class representing 1 segment of downsampling process. It consists of 2 conv layers with LeakyReLu activations
    followed by maxed pooling layer, which leads to the next such segement. Implemented as a sequential
    container added to UNet. (Things like dropout to add later)'''

    def __init__(self, channels_in, channels_out, kernel_size):
        super(Down, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, padding=get_padding_by_kernel(kernel_size)),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(),

            nn.Conv2d(channels_out, channels_out, kernel_size, padding=get_padding_by_kernel(kernel_size)),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.downsample(x)


class Up(nn.Module):
    '''Class representing 1 segment of upsampling process. It consists of 2 conv layers with LeakyReLu activations
    followed by upsampling layer, which leads to the next such segement. Implemented as a sequential
    container added to UNet. (Things like dropout to add later)'''

    def __init__(self, channels_in, channels_out, skip_channels, kernel_size, kernel_size_skip, upsampling_method):
        super(Up, self).__init__()
        self.skip = True if skip_channels > 0 else False

        self.upsample = nn.Sequential(
            nn.BatchNorm2d(channels_in + skip_channels),

            # Input channels are added to account for the concatenation with the output from skip connection
            nn.Conv2d(channels_in + skip_channels, channels_out, kernel_size, padding=get_padding_by_kernel(kernel_size)),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(),

            nn.Conv2d(channels_out, channels_out, 1),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode=upsampling_method)
        )
        if self.skip > 0:
            self.skipsample = nn.Sequential(
                nn.Conv2d(channels_in, skip_channels, kernel_size_skip, padding=get_padding_by_kernel(kernel_size_skip)),
                nn.BatchNorm2d(skip_channels),
                nn.LeakyReLU()
            )

    def forward(self, x, x_skip=None):
        if self.skip:
            # First get output from the skip connection
            out_skip = self.skipsample(x_skip)
            # Concatenate it with the input from the previous layer along the channels axis.
            concatenated = torch.cat([x, out_skip], dim=1)
            # Upsample
            x = self.upsample(concatenated)
        else:
            # Version with no skip connections
            x = self.upsample(x)
        return x

