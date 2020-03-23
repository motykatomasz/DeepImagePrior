import math
import torch
from torchvision.transforms import Compose, ToPILImage, ToTensor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_padding_by_kernel(kernel_size):
    # Assuming we are using odd kernel size
    padding_size = math.floor(kernel_size/2)
    return padding_size, padding_size


def z(shape, t='random', channels=1):
    if t =='random':
        result = 0.1 * (np.random.random((channels,) + shape))
    elif t=='meshgrid':
        result = np.zeros((2,) + shape)
        result[0, :, :], result[1, :, :] = np.meshgrid(
            np.linspace(0,1,shape[0]),
            np.linspace(0,1,shape[1])
        )
    else:
        raise KeyError("z(shape, t=%s) is not defined".format(t))

    tensor = torch.from_numpy(result)

    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor.view((1,)+(tensor.shape)).float()

def image_to_tensor(img):
    # accept a file path to an image, return a torch tensor
    pil_to_tensor = Compose([ToTensor()])
    tensor = pil_to_tensor(img)

    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor.view((1,)+(tensor.shape))


def tensor_to_image(tensor):
    # accept a torch tensor, convert it to an image at a certain path
    tensor = tensor.view(tensor.shape[1:])
    if torch.cuda.is_available():
        tensor = tensor.cpu()
    tensor_to_pil = Compose([ToPILImage()])
    pil = tensor_to_pil(tensor)
    return pil


def imshow(img):
    plt.imshow(img)
    plt.show()