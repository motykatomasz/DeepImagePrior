import math


def get_padding_by_kernel(kernel_size):
    # Assuming we are using odd kernel size
    padding_size = math.floor(kernel_size/2)
    return padding_size, padding_size


def z(shape, t='random', channels=1):
    if t=='random':
        return 0.1 * np.random.random((1,channels) + shape)
    if t=='meshgrid':
        result = np.zeros((1,2) + shape)
        result[0, 0, :, :], result[0, 1, :, :] = np.meshgrid(
            np.linspace(0,1,shape[0]),
            np.linspace(0,1,shape[1])
        )
        return result