# Deep Image Prior <br> Reproducibility Project

## Introduction
This blog post is for the reproducibility project for the TU Delft Deep Learning course. What we are going to attempt in this blog post is to reproduce the experiments and the results from the paper  "Deep Image Prior"[[1]](#citation-1) without running or consulting its available code.

## Problems tackled by the paper
The problems tackled by the paper are problem of image restoration. Some example of image restoration tasks are for example:
* Image denoising ![](https://dmitryulyanov.github.io/assets/deep-image-prior/teaser/cropped_02.png)
* Image superresolution ![](https://dmitryulyanov.github.io/assets/deep-image-prior/teaser/cropped_01.png)
* Image inpainting ![](https://dmitryulyanov.github.io/assets/deep-image-prior/teaser/cropped_09.png)

## What is Deep Image Prior
In the paper, the authors argue that a great
deal of image statistics are captured by the structure of
a convolutional image generator independent of learning.
What it means is that we can train the generator netowrk on a single degraded image, instead of large dataset of example images,
to reconstruct the image. In this scheme, the network weights serve
as a parametrization of the restored image.  


## How does it work?

Lets assume that our image x is under following process:

<inlineMath> x \Rightarrow Degradation \Rightarrow  \hat{x} \Rightarrow  Restoration \Rightarrow x^{*} </inlineMath>

Our goal is to find <inlineMath> x^{*}</inlineMath>.
We can do that by finding the MAP estimate of our posterior distribution of clean images:

```katex {evaluate: true}
MAP: x^{*} = argmax_{x} p(x|\hat{x})
```

As it is usually the case, obtaining posterior distribution <inlineMath>p(x|\hat{x})</inlineMath> is intractable. We can rewrite the equation using Bayes theorem:

```katex {evaluate: true}
p(x|\hat{x}) = \frac{p(\hat{x}|x)p(x)}{p(\hat{x})} \sim p(\hat{x}|x)p(x)
```


## Method
This approach exploits the fact that structure of a generator network are surjective mapping of <InlineMath>g:0 \to x</InlineMath>, hence the formula for optimization task <InlineMath>min_x E(x;x_0) + R(x)</InlineMath> becomes <InlineMath>min_x E(g(0);x_0) + R(g(0))</InlineMath>. Furthermore, if we select a good mapping <InlineMath>g</InlineMath>, but adjusting network hyperparameters, we could get rid of prior term and utilise a randomly initialized function as fixed input and learn from corrupted image the network parameters <InlineMath>min_z E(f(z); x_0)</InlineMath>. This parameterization network prefers naturally looking images over noise and descends more quickly in the optimization process, so the generator network provides a prior that corresponds to set of images that can be produced by the network with parameters optimized.


## Developing the Network from the paper
The main paper does not contain the structure of the network they used. Luckly, the authors of the paper provided a supplementary material document. In there they describe the netwrok they used and the hyper-parameters. In the suplementary material, they also provide a diagram with the structure of the network, as you can see in Figure [1](#figure-1), where <InlineMath>n_d[i]</InlineMath> and <InlineMath>k_d[i]</InlineMath> are respectivelly the number of filters and the kernel size of the convolutional layers of the downsampling connection <InlineMath>d_i</InlineMath>. In the same fashion <InlineMath>n_s[i]</InlineMath> and <InlineMath>k_s[i]</InlineMath> are respectivelly the number of filters and the kernel size of skip connection <InlineMath>s_i</InlineMath> and <InlineMath>n_u[i]</InlineMath> and <InlineMath>k_u[i]</InlineMath> are respectivelly the number of filters and the kernel size of upsampling connection <InlineMath>u_i</InlineMath>.

<figure id="figure-1">
  <img src="./images/network_structure.png">
  <figcaption>Figure 1 - Network Structure</figcaption>
</figure> 

The supplementary material state that the downsampling procedure they used was done by the stride implementation of the convolution, but they also state that they got a similar result with average/max pooling and downsampling with Lanczos kernel. In out implementation we decided to use max pooling. The upsampling operation is dependant on the application, but in the possible upsampling operations are nearest and bilinear. 
Our implementation of <InlineMath>d_i</InlineMath> is the following where `channels_out` is <InlineMath>n_d[i]</InlineMath> and `kernel_size` is <InlineMath>k_d[i]</InlineMath>

```python  
class Down(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, activation=nn.LeakyReLU()):
        super(Down, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, padding=get_padding_by_kernel(kernel_size)),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(channels_out),
            activation,

            nn.Conv2d(channels_out, channels_out, kernel_size, padding=get_padding_by_kernel(kernel_size)),
            nn.BatchNorm2d(channels_out),
            activation,
        )

    def forward(self, x):
        return self.downsample(x)
```
Our implementation of <InlineMath>s_i</InlineMath> where `channels_out` is <InlineMath>n_s[i]</InlineMath> and `kernel_size` is <InlineMath>k_s[i]</InlineMath>
```python
class Skip(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, activation=nn.LeakyReLU()):
        super(Skip, self).__init__()

        self.skipsample = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, padding=get_padding_by_kernel(kernel_size)),
            nn.BatchNorm2d(channels_out),
            activation,
        )

    def forward(self, x):
        return self.skipsample(x)
```
Our implementation of <InlineMath>u_i</InlineMath> where `channels_out` is <InlineMath>n_u[i]</InlineMath> and `kernel_size` is <InlineMath>k_u[i]</InlineMath>
```python
class Up(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size, upsampling_method, activation=nn.LeakyReLU()):
        super(Up, self).__init__()

        self.upsample = nn.Sequential(
            nn.BatchNorm2d(channels_in),

            nn.Conv2d(channels_in, channels_out, kernel_size, padding=get_padding_by_kernel(kernel_size)),
            nn.BatchNorm2d(channels_out),
            activation,

            nn.Conv2d(channels_out, channels_out, 1),
            nn.BatchNorm2d(channels_out),
            activation,

            nn.Upsample(scale_factor=2, mode=upsampling_method)
        )

    def forward(self, x):
        return self.upsample(x)
```
Because we are going are tackling the problem of image reconstruction, we are going to use the hyperparameters provided by the supplementary material for the image reconstruction problem which are the following:

```katex {evaluate: true}
z \in \R^{32xWxH} \sim U(0,\frac{1}{10})\\
n_u = n_d = [128, 128, 128, 128, 128] \\
k_u = k_d = [3, 3, 3, 3, 3] \\
n_s = [4, 4, 4, 4, 4] \\
k_s = [1, 1, 1, 1, 1] \\
\sigma_p = \frac{1}{30} \\
\text{num\_iter} = 11000 \\ 
\text{LR} = 0.001 \\ 
\text{upsampling} = \text{bilinear}
```

### Peculiarities From The Network Structure
Firstly as you can see from Figure [1](#figure-1), the last operation from the network is the upsampling procedure. Because the upsampling procedure is either bilinear or nearest, the resulting image is blury regardless of the network input and the network weights.  
Secondly another peculiarity is that in the case of image reconstruction, <InlineMath>n_s[5]=4</InlineMath>, which means that the number of filters and consecuently the number of channels of the output image is <InlineMath>4</InlineMath>. The paper in the case of image reconstruction experiments with grayscale images, therefore the output image should have <InlineMath>1</InlineMath> channel in total and not <InlineMath>4</InlineMath>.  
Thirdly, because the last activation function is `Leaky ReLu` the range of possible  pixel values of the resulting image is <InlineMath>(-\infin,\infin)</InlineMath> instead of <InlineMath>[0, 1]</InlineMath>.  
Forthly, in the case of the hyperparameters provided for large hole inpainting <InlineMath>n_s = [0, 0, 0, 0, 0, 0] </InlineMath> and <InlineMath> k_s=
[\text{NA}, \text{NA}, \text{NA}, \text{NA}, \text{NA}, \text{NA}]</InlineMath> which means that skip connections are omitted. But given the structure given in Figure [1](#figure-1) the only connection between the encoder and the decoder are the skip connections, which makes omitting of all the skip connections not possible.
### Addressing The Peculiarities
To solve the peculiarities, we added components from the original U-Net[[2]](#citation-2) architecture.  
To make sure that the output image, is not blury, has the right ammount of channels and has possible pixel values only within the range <InlineMath>[0, 1]</InlineMath> we added a convolutional layer  with a sigmoid activation function after the last upsampling in the same way it was done in the original U-Net architecture.  The implementation of the last layer is:
```python
class OutConv(nn.Module):
    def __init__(self, channels_in, channels_out=3):
        super(OutConv, self).__init__()

        self.outconv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.outconv(x)
```
To make sure that the encoder and the decoder are connected even without the skip connections, we added layers between the encoder's last layer and the decoder's first layer the same way it was done in the original U-Net architecture.
```python
class Connect(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, upsampling_method, activation=nn.LeakyReLU()):
        super(Connect, self).__init__()

        self.connect = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(channels_in, channels_out, kernel_size, padding=get_padding_by_kernel(kernel_size)),
            nn.BatchNorm2d(channels_out),
            activation,

            nn.Conv2d(channels_out, channels_out, kernel_size, padding=get_padding_by_kernel(kernel_size)),
            nn.BatchNorm2d(channels_out),
            activation,

            nn.Upsample(scale_factor=2, mode=upsampling_method)
        )

    def forward(self, x):
        return self.connect(x)
```
### Putting it all together

The architecture of the entire network is the following where:

* `config["channels_down"]` is <inlineMath>n_d</inlineMath> 
* `config["channels_up"]` is <inlineMath>n_u</inlineMath>, 
* `config["channels_skip"]` is <inlineMath>n_s</inlineMath>
* `config["kernels_down"]` is <inlineMath>k_d</inlineMath> 
* `config["kernels_up"]` is <inlineMath>k_u</inlineMath>, 
* `config["kernels_skip"]` is <inlineMath>k_s</inlineMath>

<collapse>

```python
class UNet(nn.Module):
    def __init__(self, channels, out_channels, config):
        super(UNet, self).__init__()
        self.channels_down = config["channels_down"]
        self.channels_up = config["channels_up"]
        self.channels_skip = config["channels_skip"]

        self.kernels_down = config["kernels_down"]
        self.kernels_up = config["kernels_up"]
        self.kernels_skip = config["kernels_skip"]

        self.upsampling_method = config["upsampling_method"]

        self.down = nn.ModuleList()
        for i in range(len(self.channels_down)):
            self.down.append(
                Down(
                    self.channels_down[i - 1] if i > 0 else channels,
                    self.channels_down[i],
                    self.kernels_down[i]
                )
            )

        self.skip = nn.ModuleList()
        for i in range(len(self.channels_skip)):
            if self.channels_skip[i] > 0:
                self.skip.append(
                    Skip(
                        self.channels_down[i],
                        self.channels_skip[i],
                        self.kernels_skip[i]
                    )
                )
            else:
                self.skip.append(None)

        self.connect_layer = Connect(
            self.channels_down[-1],
            self.channels_up[-1],
            self.kernels_down[-1],
            self.upsampling_method
        )

        self.up = nn.ModuleList()
        for i in range(len(self.channels_up)):
            self.up.append(
                Up(
                    self.channels_up[i] + self.channels_skip[i],
                    self.channels_up[i - 1] if i > 0 else self.channels_up[i],
                    self.kernels_up[i],
                    self.upsampling_method
                )
            )

        self.out_conv = OutConv(self.channels_up[0], out_channels)

    def forward(self, x):
        out = x
        x_downsampled = []
        for i in range(len(self.channels_down)):
            out = self.down[i](out)
            x_downsampled.append(out)

        out = self.connect_layer(out)

        for i in reversed(range(len(self.channels_up))):
            if self.channels_skip[i] > 0:
                out = self.up[i](torch.cat([out, self.skip[i](x_downsampled[i])], dim=1))
            else:
                out = self.up[i](out)

        return self.out_conv(out)
```
</collapse>

## Learning process
As described from the paper, we used Adam optimizer with  learning rate <inlineMath>0.0001</inlineMath> for <inlineMath>11000</inlineMath> iterations.  
In the supplementary material, they described how the optimization process destabilizes for low values of the loss function, and there after the loss function increases at consequent iterations of the optimization process. The approach describe in the supplementary material to remedy this problem was to check when the loss would be noticibly greater then its value in the previouse iteration and restore its weights to the values from the previouse iteration. We noticed that this approach does not prevent the optimization process from restabilizing afterward when the weights are restored, therefore we also diminished the learning rate to <inlineMath>LR' = 0.9LR</inlineMath> whenever we restore the weights.  
In the following plot you can see the learning process and how the change in the learning rate stabilizes the optimization process.  
{selector}
{comparison}
{plot}


## Experimental Results
### PSNR per picture

|              PSNR| Barbara | Boat  | House | Lena  | Peppers | C.man | Couple | Finger | Hill  | Man   | Montage |
| ---------------- | ------- | ----- | ----- | ----- | ------- | ----- | ------ | ------ | ----- | ----- | ------- |
| Ours             | 24.14   | 28.19 | 32.64 | 32.71 | 26.44   | 26.19 | 28.08  | 27.10  | 29.09 | 29.62 | 29.62   |
| Deep Image Prior | 32.22   | 33.06 | 39.16 | 36.16 | 33.05   | 29.8  | 32.52  | 32.84  | 32.77 | 32.20 | 34.54   |


As you can see from the table, we did not reach the performance from the main paper. This might be due to the wrong assumptions taken in the network architecture.
### References

<div id="citation-1"><strong>[1]</strong>: Ulyanov D, Vedaldi A, Lempitsky V. Deep image prior. InProceedings of  the IEEE Conference on Computer Vision and Pattern Recognition 2018 (pp. 9446-9454).</div>

<div id="citation-2"><strong>[2]</strong>: Li X, Chen H, Qi X, Dou Q, Fu CW, Heng PA. H-DenseUNet: hybrid densely connected UNet for liver and tumor segmentation from CT volumes. IEEE transactions on medical imaging. 2018 Jun 11;37(12):2663-74.</div>
