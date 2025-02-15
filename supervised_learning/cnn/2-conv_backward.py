#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """documentation documentation
    documentation documentation"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    if padding == 'valid':
        ph, pw = 0, 0
    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph),
                                    (pw, pw), (0, 0)), 'constant')
    dA_prev = np.zeros(A_prev_padded.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    dA_prev[i, h * sh: h * sh + kh, w * sw: w * sw + kw, :]+= W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += A_prev_padded[i, h * sh: h * sh + kh, w * sw: w * sw + kw, :] * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
    if padding == 'same':
        dA_prev = dA_prev[:, ph: -ph, pw: -pw, :]
    return dA_prev, dW, db
