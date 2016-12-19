import math

import numpy as np
import PIL.Image
import scipy.signal
import click


def dist(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


def normalize(array, new_min, new_max, ignore_first_max = False):
    min = array.min()
    max = array.max()

    # Usa o segundo maior valor como o máximo
    if ignore_first_max:
        max = np.partition(array, -2, None)[-2]

    return (array - min) * ((new_max - new_min) / (max - min)) + new_min


# Reference: http://stackoverflow.com/a/36961324
def matrix_spiral(A):
    B = []
    while(A.size):
        B.append(A[0][::-1])
        A = A[1:][::-1].T

    return np.concatenate(B)


def base_spiral(nrow, ncol):
    return matrix_spiral(np.arange(nrow*ncol).reshape(nrow, ncol))[::-1]


def matrix_to_spiral(A):
    return A.flat[base_spiral(*A.shape)]


def matrix_from_spiral(A, nrow, ncol):
    B = np.empty((nrow, ncol), dtype=A.dtype)
    B.flat[base_spiral(nrow, ncol)] = A
    return B
