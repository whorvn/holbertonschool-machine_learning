#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


import numpy as np



def pool(images, kernel_shape, stride, mode='max'):
    """documentation documentation
    documentation documentation"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    output_h = int((h - kh) / sh) + 1
    output_w = int((w - kw) / sw) + 1
    output = np.zeros((m, output_h, output_w, c))
    for i in range(output_h):
        for j in range(output_w):
            if mode == 'max':
                output[:, i, j] = images[:,
                                         i * sh: i * sh + kh,
                                         j * sw: j * sw + kw
                                         ].max(axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j] = images[:,
                                         i * sh: i * sh + kh,
                                         j * sw: j * sw + kw
                                         ].mean(axis=(1, 2))
    return output
