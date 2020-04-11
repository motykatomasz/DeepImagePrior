#first item in channels_down is number of input channels
#last item in channels_up is number of output channels

denoisingSettings = {
    "channels_down": [8, 16, 32, 64, 128],
    "channels_up": [8, 16, 32, 64, 128],
    "channels_skip": [0, 0, 0, 4, 4],
    "kernels_down": [3, 3, 3, 3, 3],
    "kernels_up": [3, 3, 3, 3, 3],
    "kernels_skip": [0, 0, 0, 1, 1],
    "upsampling_method": "bilinear",
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

largeHoleInpaintingSettingsLibrary = {
    "channels_down": [16, 32, 64, 128, 128, 128],
    "channels_up": [16, 32, 64, 128, 128, 128],
    "channels_skip": [0, 0, 0, 0, 0, 0],
    "kernels_down": [3, 3, 3, 3, 3, 3],
    "kernels_up": [5, 5, 5, 5, 5, 5],
    "kernels_skip": [0, 0, 0, 0, 0, 0],
    "upsampling_method": "nearest",
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