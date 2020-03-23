import torch
from PIL import Image
from numpy import asarray
from models.unet import UNet
import torch.optim as optim
from models.utils import z, imshow, image_to_tensor, tensor_to_image

img_path = "data/kate.png"
img = Image.open(img_path)
imshow(asarray(img))

mask_path = "data/kate_mask.png"
mask = Image.open(mask_path)
imshow(asarray(mask))

x = image_to_tensor(img)
mask = image_to_tensor(mask)

net = UNet()

if torch.cuda.is_available():
    net = net.cuda()

optimizer = optim.Adam(net.parameters(), lr=0.001)

# Num of iters for training
num_iters = 1000

# Num of iters when to save image
save_frequency = 250

#Since we only have 1 image to train on, we set zero_gradienet once at the beginning
optimizer.zero_grad()

z0 = z(shape=(img.height, img.width), channels=3)

for i in range(num_iters):
    output = net(z0)

    # Optimizer
    loss = torch.sum(torch.mul((output - x), mask)**2)
    loss.backward()
    optimizer.step()

    print('Step :{}, Loss: {}'.format(i, loss.data.cpu()))

    if i % save_frequency == 0:
        out_img = tensor_to_image(output)
        imshow(asarray(out_img))
        print('OUTPUT IMAGE')

