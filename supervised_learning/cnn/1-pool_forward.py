#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """documentation documentation
    documentation documentation"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h = int((h_prev - kh) / sh) + 1
    w = int((w_prev - kw) / sw) + 1
    output = np.zeros((m, h, w, c_prev))
    for i in range(h):
        for j in range(w):
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2)
                )
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(
                    A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2)
                )
    return output
