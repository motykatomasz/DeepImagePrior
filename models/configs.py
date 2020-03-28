#first item in channels_down is number of input channels
#last item in channels_up is number of output channels

inpaintingSettings = {
    "channels_down": [3, 16, 32, 64, 128, 128, 128],
    "channels_up": [128, 128, 128, 64, 32, 16, 3],
    "channels_skip": [0, 0, 0, 0, 0, 0],
    "kernels_down": [3, 3, 3, 3, 3, 3],
    "kernels_up": [5, 5, 5, 5, 5, 5],
    "kernels_skip": [0, 0, 0, 0, 0, 0],
    "upsampling_method": "nearest",
    "activation": "sigmoid",
}

superresolutionSettings = {
    "channels_down": [32, 128,128, 128, 128, 128],
    "channels_up": [128, 128, 128, 128, 128, 3],
    "channels_skip": [4, 4, 4, 4, 4],
    "kernels_down": [3, 3, 3, 3, 3],
    "kernels_up": [3, 3, 3, 3, 3],
    "kernels_skip": [1, 1, 1, 1, 1],
    "upsampling_method": "bilinear",
    "activation": "relu",
}


textInpaintingSettings = {
    "channels_down": [128, 128, 128, 128, 128],
    "channels_up": [128, 128, 128, 128, 128],
    "channels_skip": [4, 4, 4, 4, 4],
    "kernels_down": [3, 3, 3, 3, 3],
    "kernels_up": [3, 3, 3, 3, 3],
    "kernels_skip": [1, 1, 1, 1, 1],
    "upsampling_method": "bilinear",
}

testSettings = {
    "channels_down": [3, 16, 32, 64, 128],
    "channels_up": [128, 64, 32, 16, 3],
    "channels_skip": [0, 0, 0, 0],
    "kernels_down": [3, 3, 3, 3],
    "kernels_up": [5, 5, 5, 5],
    "kernels_skip": [0, 0, 0, 0],
    "upsampling_method": "nearest",
    "activation": "sigmoid",
}