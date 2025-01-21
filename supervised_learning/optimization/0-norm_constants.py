#!/usr/bin/env python3
"""
Defines a function that builds a neural network
using Keras library
"""


import numpy as np


def normalization_constants(X):
    """
    calculates the normalization (standardization) constants of a matrix
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
