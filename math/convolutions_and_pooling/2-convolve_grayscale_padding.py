#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """documentation documentation
    documentation documentation"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    output_h = h - kh + (2 * ph) + 1
    output_w = w - kw + (2 * pw) + 1
    output = np.zeros((m, output_h, output_w))
    image = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (image[:, i: i + kh, j: j + kw] * kernel
                               ).sum(axis=(1, 2))
    return output
