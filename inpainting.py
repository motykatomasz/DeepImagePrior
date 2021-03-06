import torch
import copy
from PIL import Image
from numpy import asarray
from models.unet import UNet
import torch.optim as optim
from models.utils import z, imshow, image_to_tensor, tensor_to_image
from models.configs import textInpaintingSettings

img_path = "data/inpainting/kate.png"
img = Image.open(img_path)
imshow(asarray(img))

mask_path = "data/inpainting/kate_mask.png"
mask = Image.open(mask_path)
imshow(asarray(mask))

x = image_to_tensor(img)
mask = image_to_tensor(mask)

img_masked = x * mask

net = UNet(32, textInpaintingSettings)

if torch.cuda.is_available():
    net = net.cuda()

mse = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Num of iters for training
num_iters = 5000

# Num of iters when to save image
save_frequency = 50

z0 = z(shape=(img.height, img.width), channels=32)
z0_saved = z0.detach().clone()
noise = z0.detach().clone()
z0_noise_std = 0.03

last_lost = None
last_model_state = net.state_dict()

for i in range(num_iters):

    if z0_noise_std > 0:
        z0 = z0_saved + (noise.normal_() * z0_noise_std)

    optimizer.zero_grad()
    output = net(z0)

    # Optimizer
    # loss = torch.sum(torch.mul((output - x), mask)**2)
    loss = mse(output * mask, x * mask)
    # loss = mse(output * mask, img_masked)
    loss.backward()
    optimizer.step()

    cpu_loss = loss.data.cpu()
    print('Step :{}, Loss: {}'.format(i, cpu_loss))
    if i % save_frequency == 0:
        out_img = tensor_to_image(output)
        imshow(asarray(out_img))
        print('OUTPUT IMAGE')

    if last_lost is None:
        last_lost = cpu_loss

    if cpu_loss - last_lost > 50:
        net.load_state_dict(last_model_state)

    last_model_state = copy.deepcopy(net.state_dict())
    last_lost = cpu_loss

