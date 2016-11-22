import math

import numpy as np
import PIL.Image
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
