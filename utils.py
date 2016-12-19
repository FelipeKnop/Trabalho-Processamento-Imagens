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

# Reference: http://stackoverflow.com/a/24892274
def rl_encode(A):
    # A = np.array([1,1,1,1,0,8,1,1,1,1,1,0,2,2,2,9,9,9,9,9,9,0,1,9,9,0,9])
    iszero = np.concatenate(([0], np.equal(A, 0), [0]))
    absdiff = np.abs(np.diff(iszero))

    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)[::-1]

    for (begin, end) in ranges:
        while(end - begin > 255):
            A = np.concatenate((A[:(end - 255)], [0, 255], A[end:]))
            end -= 255

        A = np.concatenate((A[:begin], [0, end - begin], A[end:]))


    click.echo(A)
    return A

def rl_decode(A):
    iszero = np.where(A[:-1] == 0)[0][::-1]
    click.echo(A)

    for begin in iszero:
        # click.echo(A)
        A = np.concatenate((A[:begin], np.full(A[begin + 1], A[begin], dtype=A.dtype), A[begin + 2:]))

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
