import torch.nn as nn


class Down(nn.Module):
    '''Class representing 1 segment of downsampling process. It consists of 2 conv layers with LeakyReLu activations
    followed by maxed pooling layer, which leads to the next such segement. Implemented as a sequential
    container added to UNet. (Things like dropout to add later)'''

    def __init__(self, channels_in, channels_out, LR):
        super(Down, self).__init__()
        self.downsample = nn.Sequential(
<<<<<<< HEAD
            nn.Conv2d(channels_in, channels_in, 3, padding="same"),
=======
            # Padding yet to figure out
            nn.Conv2d(channels_in, channels_out, 3, padding=(1, 1)),
>>>>>>> 86a05ef4e4cebe41ca98d9b3443742d354eb0d1c
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
<<<<<<< HEAD
    '''Class representing 1 segment of downsampling process. It consists of 2 conv layers with ReLu activations
    followed by maxed pooling layer, which leads to the next such segement. Implemented as a sequential
=======
    '''Class representing 1 segment of upsampling process. It consists of 2 conv layers with LeakyReLu activations
    followed by upsampling layer, which leads to the next such segement. Implemented as a sequential
>>>>>>> 86a05ef4e4cebe41ca98d9b3443742d354eb0d1c
    container added to UNet. (Things like dropout to add later)'''

    def __init__(self, channels_in, channels_out, kernel_size, LR):
        super(Up, self).__init__()
        self.upsample = nn.Sequential(
            nn.BatchNorm2d(channels_in),

<<<<<<< HEAD
            nn.Conv2d(channels_in, channels_in, kernel_size, padding_mode='reflect'),
            nn.BatchNorm2d(channels_in),
            nn.LeakyReLU(negative_slope=LR),

            nn.Conv2d(channels_in, channels_in, kernel_size, padding_mode='reflect'),
            nn.BatchNorm2d(channels_in),
            nn.LeakyReLU(negative_slope=LR),

            nn.Upsample(channels_out, mode="bilinear")
        )

    def forward(self, x):
        return self.upsample
=======
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
>>>>>>> 86a05ef4e4cebe41ca98d9b3443742d354eb0d1c

#TODO Possibly
class Skip(nn.Module):
    ...