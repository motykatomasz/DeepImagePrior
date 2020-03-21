import torch.nn as nn
import torch
from PIL import Image
from numpy import asarray
from models.unet import UNet
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, ToPILImage, ToTensor


def imshow(img):
    plt.imshow(img)
    plt.show()


dtype = torch.cuda.FloatTensor

use_gpu = torch.cuda.is_available()
print(use_gpu)

img_path = "../data/kate.png"
img = Image.open(img_path)
imshow(asarray(img))

mask_path = "../data/kate_mask.png"
mask = Image.open(mask_path)
imshow(asarray(mask))


x = TF.to_tensor(img)
x.unsqueeze_(0)
x = x.cuda()

net = UNet()

if use_gpu:
    net = net.cuda()

# Test if network forwards the input
out = net(x)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_iters = 1000

# Num of iters when to save image
save_frequency = 250

#blackout pixels' proportion in percentage
prop = 0.5

#Since we only have 1 image to train on, we set zero_gradienet once at the beginning
optimizer.zero_grad()

# Network Model
net = UNet()


def image_to_tensor(filepath=img_path):
    # accept a file path to an image, return a torch tensor
    pil = Image.open(filepath)
    pil_to_tensor = Compose([ToTensor()])
    if use_gpu:
        tensor = pil_to_tensor(pil).cuda()
    else:
        tensor = pil_to_tensor(pil)
    return tensor.view([1]+list(tensor.shape))


def tensor_to_image(tensor, filename):
    # accept a torch tensor, convert it to an image at a certain path
    tensor = tensor.view(tensor.shape[1:])
    if use_gpu:
        tensor = tensor.cpu()
    tensor_to_pil = Compose([ToPILImage()])
    pil = tensor_to_pil(tensor)
    pil.save(filename)


def zero_out_pixels(tensor, prop=prop):
    # function which zeros out a random proportion of pixels from an image tensor.
    if use_gpu:
        mask = torch.rand([1]+[1] + list(tensor.shape[2:])).cuda()
    else:
        mask = torch.rand([1]+[1] + list(tensor.shape[2:]))
    mask[mask<prop] = 0
    mask[mask!=0] = 1
    mask = mask.repeat(1,3,1,1)
    deconstructed = tensor * mask
    return mask, deconstructed


for iter in range(num_iters):

    output = net(x)

    mask, deconstructed = zero_out_pixels(image_to_tensor(img_path))
    mask_output = output * mask

    # Optimizer
    loss = torch.sum((mask_output - deconstructed)**2)
    loss.backward()
    optimizer.step()

    print('Step :{}, Loss: {}'.format(iter, loss.data.cpu()))
    
    #ToDo
    if num_iters % save_frequency == 0:
        print('OUTPUT IMAGE')
