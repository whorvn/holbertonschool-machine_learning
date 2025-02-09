#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """documentation documentation
    documentation documentation"""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    output_h = int((h - kh + (2 * ph)) / sh) + 1
    output_w = int((w - kw + (2 * pw)) / sw) + 1
    output = np.zeros((m, output_h, output_w, nc))
    image = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                output[:, i, j, k] = (image[:, i * sh: i * sh + kh,
                                     j * sw: j * sw + kw] * kernels[:, :, :, k]
                                      ).sum(axis=(1, 2, 3))
    return output
