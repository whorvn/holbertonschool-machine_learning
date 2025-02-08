#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """documentation documentation
    documentation documentation"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = kh // 2
    pw = kw // 2
    output_h = h
    output_w = w
    output = np.zeros((m, output_h, output_w))
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (images_padded[:, i: i + kh, j: j + kw] * kernel
                               ).sum(axis=(1, 2))
    return output
