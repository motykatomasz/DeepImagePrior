import math
import torch
from torchvision.transforms import Compose, ToPILImage, ToTensor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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

def psnr(input, output):
    # Peak Signal to Noise Ratio calculated using either mean square error or root mean square error
    # Compares error between Input matrix with Output matrix of pixels of their respective images.
    # input = input image tensor, output = output image tensor

    mse = torch.mean((input - output)**(2))
    psnr = 20 * torch.log(1 / torch.sqrt(mse))
    return psnr


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

def image_to_numpy(img_PIL):
    ar = np.array(img_PIL).transpose(2, 0, 1)
    return ar.astype(np.float32) / 255.


def numpy_to_tensor(img_np):
    tensor = torch.from_numpy(img_np)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor.view((1,)+(tensor.shape)).float()


def crop_image(img, d=32):
    # make image dimensions divisible by d
    # needed so that output of the network keeps the dimensions
    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2),
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image."""
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    return img_noisy_np


def imshow(img):
    plt.imshow(img)
    plt.show()