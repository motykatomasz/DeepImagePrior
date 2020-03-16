import torch.nn as nn


class Down(nn.Module):
    '''Class representing 1 segment of downsampling process. It consists of 2 conv layers with ReLu activations
    followed by maxed pooling layer, which leads to the next such segement. Implemented as a sequential
    container added to UNet. (Things like dropout to add later)'''

    def __init__(self, channels_in, channels_out, LR):
        super(Down, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding="same"),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(negative_slope=LR),
            nn.Conv2d(channels_out, channels_out, 3, padding="same"),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(negative_slope=LR)
        )

    def forward(self, x):
        return self.downsample

#TODO
class Up(nn.Module):
    ...

#TODO Possibly
class Skip(nn.Module):
    ...