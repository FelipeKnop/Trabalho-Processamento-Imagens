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
