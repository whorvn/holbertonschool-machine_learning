#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """documentation documentation
    documentation documentation"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (images[:, i: i + kh, j: j + kw] * kernel
                               ).sum(axis=(1, 2))
    return output
