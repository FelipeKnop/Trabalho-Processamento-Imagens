import math

import numpy as np
import PIL.Image
import scipy.signal
import click

import matplotlib.pyplot as plt

def dist(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


def normalize(array, new_min, new_max, ignore_first_max = False):
    min = array.min()
    max = array.max()

    # Usa o segundo maior valor como o máximo
    if ignore_first_max:
        max = np.partition(array, -2, None)[-2]

    return (array - min) * ((new_max - new_min) / (max - min)) + new_min


# Reference: http://stackoverflow.com/a/29216609
def quantization_matrix(shape, Q=50):
    temp_m = np.arange(0, shape[0]) - int(shape[0] / 2)
    temp_n = np.arange(0, shape[1]) - int(shape[1] / 2)
    xx, yy = np.meshgrid(temp_m, temp_n)
    base_matrix = ((xx**2 + yy*2) * 8 + 128).astype('int')

    # base_matrix = np.array([[82,  82,  82,  82,  82,  82,  82,  82],
    #                         [82,  53,  53,  53,  53,  53,  53,  82],
    #                         [82,  53,  22,  22,  22,  22,  53,  82],
    #                         [82,  53,  22,  10,  10,  22,  53,  82],
    #                         [82,  53,  22,  10,  10,  22,  53,  82],
    #                         [82,  53,  22,  22,  22,  22,  53,  82],
    #                         [82,  53,  53,  53,  53,  53,  53,  82],
    #                         [82,  82,  82,  82,  82,  82,  82,  82]])
    # base_matrix = np.array([ 16,  11,  12,  14,  12,  10,  16,  14,
    #                          13,  14,  18,  17,  16,  19,  24,  40,
    #                          26,  24,  22,  22,  24,  49,  35,  37,
    #                          29,  40,  58,  51,  61,  60,  57,  51,
    #                          56,  55,  64,  72,  92,  78,  64,  68,
    #                          87,  69,  55,  56,  80, 109,  81,  87,
    #                          95,  98, 103, 104, 103,  62,  77, 113,
    #                         121, 112, 100, 120,  92, 101, 103,  99])
    # base_matrix = np.linspace(0.7, 0.1, A.shape[0]) * (np.abs(A).max() / 255)

    if Q < 50:
        S = 5000/(Q+1)
    else:
        S = 200 - 2*Q

    q_matrix = np.floor((S * base_matrix + 50) / 100).astype('int')

    return q_matrix


# Reference: http://stackoverflow.com/a/16715845
def submatrices(A, nrow, ncol, padding=True):
    if padding:
        padr = (A.shape[0]-1) % nrow + 1
        padc = (A.shape[1]-1) % ncol + 1
        A = np.pad(A, ((0, nrow - padr), (0, ncol - padc)), 'edge')

    lenr = int(A.shape[0] / nrow)
    lenc = int(A.shape[1] / ncol)

    return np.array([A[i*nrow:(i + 1)*nrow, j*ncol:(j + 1)*ncol] for (i, j) in np.ndindex(lenr, lenc)]).reshape(lenr, lenc, nrow, ncol)


def delta_encode(A):
    return A - np.concatenate(([0], A[:-1]))


def delta_decode(A):
    return np.cumsum(A)


# Reference: http://stackoverflow.com/a/24892274
def rl_encode(A, value=0):
    isvalue = np.concatenate(([0], np.equal(A, value), [0]))
    absdiff = np.abs(np.diff(isvalue))

    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)[::-1]

    for (begin, end) in ranges:
        while(end - begin > 255):
            A = np.concatenate((A[:(end - 255)], np.array([0, 255], dtype=A.dtype), A[end:]))
            end -= 255

        A = np.concatenate((A[:begin], np.array([0, end - begin], dtype=A.dtype), A[end:]))

    return A


def rl_decode(A, value=0):
    isvalue = np.where(A[:-1] == 0)[0][::-1]

    for begin in isvalue:
        A = np.concatenate((A[:begin], np.full(A[begin + 1], value, dtype=A.dtype), A[begin + 2:]))

    return A


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
