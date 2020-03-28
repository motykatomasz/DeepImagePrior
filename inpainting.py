import torch
from PIL import Image
from numpy import asarray
from models.unet import UNet
import torch.optim as optim
from models.utils import z, imshow, image_to_tensor, tensor_to_image
from models.configs import inpaintingSettings

img_path = "data/inpainting/kate.png"
img = Image.open(img_path)
imshow(asarray(img))

mask_path = "data/inpainting/kate_mask.png"
mask = Image.open(mask_path)
imshow(asarray(mask))

x = image_to_tensor(img)
mask = image_to_tensor(mask)

img_masked = x * mask

net = UNet(inpaintingSettings)

if torch.cuda.is_available():
    net = net.cuda()

mse = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)

# Num of iters for training
num_iters = 5000

# Num of iters when to save image
save_frequency = 250

z0 = z(shape=(img.height, img.width), channels=3)

for i in range(num_iters):
    optimizer.zero_grad()
    output = net(z0)

    # Optimizer
    # loss = torch.sum(torch.mul((output - x), mask)**2)
    loss = mse(output * mask, x * mask)
    # loss = mse(output * mask, img_masked)
    loss.backward()
    optimizer.step()

    if i % save_frequency == 0:
        print('Step :{}, Loss: {}'.format(i, loss.data.cpu()))
        out_img = tensor_to_image(output)
        imshow(asarray(out_img))
        print('OUTPUT IMAGE')

