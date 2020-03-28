import torch
from PIL import Image
from numpy import asarray
from models.unet import UNet
import torch.optim as optim
from models.utils import z, imshow, image_to_tensor, tensor_to_image, crop_image
from models.configs import superresolutionSettings

img_path = "data/superresolution/snail.jpg"
img = Image.open(img_path)
imshow(asarray(img))

img = crop_image(img)

x = image_to_tensor(img)

net = UNet(superresolutionSettings)

if torch.cuda.is_available():
    net = net.cuda()

mse = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Num of iters for training
num_iters = 2000

# Num of iters when to save image
save_frequency = 100

z0 = z(shape=(img.height, img.width), channels=32)

for i in range(num_iters):
    optimizer.zero_grad()
    output = net(z0)

    # Optimizer
    loss = mse(output, x)
    loss.backward()
    optimizer.step()

    if i % save_frequency == 0:
        print('Step :{}, Loss: {}'.format(i, loss.data.cpu()))
        out_img = tensor_to_image(output)
        imshow(asarray(out_img))
        print('OUTPUT IMAGE')

