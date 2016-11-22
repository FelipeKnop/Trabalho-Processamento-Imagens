import math

import numpy as np
import PIL.Image
import scipy.signal
import click

# NOTE(andre:2016-11-21): É necessario usar a PIL para isso?
def convert_image(image, mode, from_mode='RGB'):
    if from_mode is not None:
        pil_image = PIL.Image.fromarray(image, from_mode)
    else:
        pil_image = PIL.Image.fromarray(image)

    return np.array(pil_image.convert(mode))


# NOTE(andre:2016-11-21): Matrix Form of 2D DFT: http://fourier.eng.hmc.edu/e101/lectures/Image_Processing/node6.html
def fft2(image):
    M, N = image.shape

    temp_m = np.arange(0, M)
    temp_n = np.arange(0, N)
    m, k = np.meshgrid(temp_m, temp_m)
    n, l = np.meshgrid(temp_n, temp_n)

    Wm = (1 / M) * np.exp(-1j * 2 * math.pi * m * k / M)
    Wn = (1 / N) * np.exp(-1j * 2 * math.pi * n * l / N)

    return np.matmul(np.matmul(Wm, image), Wn)


def ifft2(image):
    M, N = image.shape

    temp_m = np.arange(0, M)
    temp_n = np.arange(0, N)
    m, k = np.meshgrid(temp_m, temp_m)
    n, l = np.meshgrid(temp_n, temp_n)

    Wm = np.exp(1j * 2 * math.pi * m * k / M)
    Wn = np.exp(1j * 2 * math.pi * n * l / N)

    return np.matmul(np.matmul(Wm, image), Wn)

def convolve(image, axis, mode, kernel, dimension_x, dimension_y, degree, sigma):
    image = np.array(image, dtype='float64')

    if kernel == 'gaussian':
        dimension_x = 6 * sigma
        # dimension_y = 6 * sigma

        Gx = np.linspace(-int(dimension_x / 2), int(dimension_x / 2), dimension_x)
        Gx = np.exp((-(Gx ** 2) / (2 * (sigma ** 2))))
        Gx /= Gx.sum()

        # Gy = np.linspace(-int(dimension_y / 2), int(dimension_y / 2), dimension_y)
        # Gy = np.exp((-(Gy ** 2) / (2 * (sigma ** 2))))
        # Gy /= Gy.sum()

        image = scipy.ndimage.filters.convolve1d(image, Gx, axis=0, mode=mode)
        image = scipy.ndimage.filters.convolve1d(image, Gx, axis=1, mode=mode)
        # image = scipy.ndimage.filters.convolve1d(image, Gy, axis=1, mode=mode)

    elif kernel == 'prewitt':
        Px = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])
        Py = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ])

        for c in range(0, image.shape[2]):
            dx = scipy.ndimage.filters.convolve(image[:,:,c], Px, mode=mode)
            dy = scipy.ndimage.filters.convolve(image[:,:,c], Py, mode=mode)
            image[:,:,c] = np.hypot(dx, dy)
            max_value = np.max(image[:,:,c])
            image[:,:,c] *= 255.0 / max_value

    elif kernel == 'sobel':
        Sx = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        Sy = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

        for c in range(0, image.shape[2]):
            dx = scipy.ndimage.filters.convolve(image[:,:,c], Sx, mode=mode)
            dy = scipy.ndimage.filters.convolve(image[:,:,c], Sy, mode=mode)
            image[:,:,c] = np.hypot(dx, dy)
            max_value = np.max(image[:,:,c])
            image[:,:,c] *= 255.0 / max_value

    elif kernel == 'roberts':
        Rx = np.array([
            [1, 0],
            [0, -1]
        ])
        Ry = np.array([
            [0, 1],
            [-1, 0]
        ])

        for c in range(0, image.shape[2]):
            dx = scipy.ndimage.filters.convolve(image[:,:,c], Rx, mode=mode)
            dy = scipy.ndimage.filters.convolve(image[:,:,c], Ry, mode=mode)
            image[:,:,c] = np.hypot(dx, dy)
            max_value = np.max(image[:,:,c])
            image[:,:,c] *= 255.0 / max_value

    elif kernel == 'laplace':
        L = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ])

        for c in range(0, image.shape[2]):
            image[:,:,c] = scipy.ndimage.filters.convolve(image[:,:,c], L, mode=mode)
            # image[:,:,c] -= np.min(image[:,:,c])
            image[:,:,c] = np.absolute(image[:,:,c])
            max_value = np.max(image[:,:,c])
            image[:,:,c] *= 255.0 / max_value

    else:
        weights = box = np.ones(dimension_x)
        for g in range(0, degree - 1):
            weights = scipy.signal.convolve(weights, box, mode='same')

        weights = weights / weights.sum()

        if axis == 'x':
            image = scipy.ndimage.filters.convolve1d(image, weights, axis=0, mode=mode)
        elif axis == 'y':
            image = scipy.ndimage.filters.convolve1d(image, weights, axis=1, mode=mode)
        elif axis == 'xy':
            image = scipy.ndimage.filters.convolve1d(image, weights, axis=0, mode=mode)
            image = scipy.ndimage.filters.convolve1d(image, weights, axis=1, mode=mode)

    return image.astype('uint8')
