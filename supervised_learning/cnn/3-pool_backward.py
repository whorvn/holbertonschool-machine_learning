#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """documentation documentation
    documentation documentation"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    if mode == 'max':
                        A = A_prev[i, h * sh: h * sh + kh, w * sw: w * sw + kw, c]
                        mask = (A == np.max(A))
                        dA_prev[i, h * sh: h * sh + kh, w * sw: w * sw + kw, c] += dA[i, h, w, c] * mask
                    if mode == 'avg':
                        da = dA[i, h, w, c]
                        avg = da / kh / kw
                        dA_prev[i, h * sh: h * sh + kh, w * sw: w * sw + kw, c] += np.ones(kernel_shape) * avg
    return dA_prev
