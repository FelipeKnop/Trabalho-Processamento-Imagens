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


# Reference: http://stackoverflow.com/a/16715845
def submatrices(A, nrow, ncol, padding=True):
    if padding:
        padr = A.shape[0] % nrow
        padc = A.shape[1] % ncol
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
