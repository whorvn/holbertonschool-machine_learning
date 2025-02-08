#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """documentation"""
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    h = int((h + 2 * ph - kh) / sh + 1)
    w = int((w + 2 * pw - kw) / sw + 1)
    new_img = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            new_img[:, i, j] = (images[:, i * sh:i * sh + kh,
                            j * sw:j * sw + kw] * kernel).sum(axis=(1, 2, 3))

    return new_img
