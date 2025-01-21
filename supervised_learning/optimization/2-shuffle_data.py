#!/usr/bin/env python3
"""
Defines a function that builds a neural network
using Keras library
"""


import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    return X[shuffle], Y[shuffle]
