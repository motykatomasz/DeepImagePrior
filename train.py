import torch.nn as nn
import torch
from PIL import Image
from numpy import asarray
from models.unet import UNet
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


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

#Since we only have 1 image to train on, we set zero_gradienet once at the beginning
optimizer.zero_grad()

for iter in range(num_iters):
    ...
    # Training part