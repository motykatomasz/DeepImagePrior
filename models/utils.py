import math
import torch
from torchvision.transforms import Compose, ToPILImage, ToTensor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

use_gpu = torch.cuda.is_available()


def get_padding_by_kernel(kernel_size):
    # Assuming we are using odd kernel size
    padding_size = math.floor(kernel_size/2)
    return padding_size, padding_size


def z(shape, t='random', channels=1):
    if t =='random':
        return 0.1 * (np.random.random((1, channels) + shape))
    if t=='meshgrid':
        result = np.zeros((1,2) + shape)
        result[0, 0, :, :], result[0, 1, :, :] = np.meshgrid(
            np.linspace(0,1,shape[0]),
            np.linspace(0,1,shape[1])
        )
        return result


def psnr(input, output):
    # Peak Signal to Noise Ratio calculated using either mean square error or root mean square error
    # Compares error between Input matrix with Output matrix of pixels of their respective images.
    # input = input image tensor, output = output image tensor

    mse = torch.mean((input - output)**(2))
    psnr = 20 * torch.log(255 / torch.sqrt(mse))
    return psnr


def image_to_tensor(img):
    # accept a file path to an image, return a torch tensor
    pil_to_tensor = Compose([ToTensor()])
    if use_gpu:
        tensor = pil_to_tensor(img).cuda()
    else:
        tensor = pil_to_tensor(img)
    return tensor.view([1]+list(tensor.shape))


def tensor_to_image(tensor):
    # accept a torch tensor, convert it to an image at a certain path
    tensor = tensor.view(tensor.shape[1:])
    if use_gpu:
        tensor = tensor.cpu()
    tensor_to_pil = Compose([ToPILImage()])
    pil = tensor_to_pil(tensor)
    return pil


def imshow(img):
    plt.imshow(img)
    plt.show()