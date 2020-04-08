import torch
import copy
from PIL import Image
from numpy import asarray
from models.unet import UNet
import torch.optim as optim
from models.utils import z, imshow, image_to_tensor, tensor_to_image, get_noisy_image, numpy_to_tensor, image_to_numpy
from models.configs import denoisingSettings

img_path = "data/denoising/F16_GT.png"
img = Image.open(img_path)
img_np = asarray(img)
imshow(img_np)

sigma = 1./10.
noisy_img = get_noisy_image(image_to_numpy(img), sigma)

x = numpy_to_tensor(noisy_img)

imshow(asarray(tensor_to_image(x)))

net = UNet(3, denoisingSettings)

if torch.cuda.is_available():
    net = net.cuda()

mse = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Num of iters for training
num_iters = 2400

# Num of iters when to save image
save_frequency = 50

z0 = z(shape=(img.height, img.width), channels=3)

out_avg = None
z0_saved = z0.detach().clone()
noise = z0.detach().clone()
z0_noise_std = 1./30.

last_lost = None
last_model_state = net.state_dict()

for i in range(num_iters):
    optimizer.zero_grad()

    if z0_noise_std > 0:
        z0 = z0_saved + (noise.normal_() * z0_noise_std)

    output = net(z0)

    # Optimizer
    loss = mse(output, x)
    loss.backward()
    optimizer.step()

    cpu_loss = loss.data.cpu()
    if i % save_frequency == 0:
        print('Step :{}, Loss: {}'.format(i, loss.data.cpu()))
        out_img = tensor_to_image(output)
        imshow(asarray(out_img))
        print('OUTPUT IMAGE')

    if last_lost is None:
        last_lost = cpu_loss

    if cpu_loss - last_lost > 50:
        net.load_state_dict(last_model_state)

    last_model_state = copy.deepcopy(net.state_dict())
    last_lost = cpu_loss

