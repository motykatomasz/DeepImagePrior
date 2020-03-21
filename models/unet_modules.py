import torch.nn as nn
import numpy as np


class Down(nn.Module):
    '''Class representing 1 segment of downsampling process. It consists of 2 conv layers with LeakyReLu activations
    followed by maxed pooling layer, which leads to the next such segement. Implemented as a sequential
    container added to UNet. (Things like dropout to add later)'''

    def __init__(self, channels_in, channels_out, LR):
        super(Down, self).__init__()
        self.downsample = nn.Sequential(
            # Padding yet to figure out
            nn.Conv2d(channels_in, channels_out, 3, padding=(1, 1)),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(negative_slope=LR),

            nn.Conv2d(channels_out, channels_out, 3, padding=(1, 1)),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(negative_slope=LR),
        )

    def forward(self, x):
        return self.downsample(x)


class Up(nn.Module):
    '''Class representing 1 segment of upsampling process. It consists of 2 conv layers with LeakyReLu activations
    followed by upsampling layer, which leads to the next such segement. Implemented as a sequential
    container added to UNet. (Things like dropout to add later)'''

    def __init__(self, channels_in, channels_out, kernel_size, LR):
        super(Up, self).__init__()
        self.upsample = nn.Sequential(
            nn.BatchNorm2d(channels_in),

            # Padding yet to figure out
            nn.Conv2d(channels_in, channels_out, kernel_size, padding=(2, 2)),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(negative_slope=LR),

            nn.Conv2d(channels_out, channels_out, 1),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(negative_slope=LR),

            nn.Upsample(scale_factor=2, mode="nearest")
        )

    def forward(self, x, x_skip=None):
        if x_skip:
            #TODO #Here to add concatenation from downsamplig. Mind differences in network architecture in the paper and original UNet.
            ...
        else:
            # Version with no skip connections
            x = self.upsample(x)
        return x

#TODO Possibly
class Skip(nn.Module):
    ...

def z(shape, t='random', channels=1):
    if t=='random':
        return 0.1 * np.random.random((1,channels) + shape)
    if t=='meshgrid':
        result = np.zeros((1,2) + shape)
        result[0, 0, :, :], result[0, 1, :, :] = np.meshgrid(
            np.linspace(0,1,shape[0]),
            np.linspace(0,1,shape[1])
        )
        return result