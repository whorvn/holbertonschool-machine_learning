#!/usr/bin/env python3
"""
Defines a function that builds a neural network
using Keras library
"""


import tensorflow.keras as K


def normalization_constants(X):
    """
    calculates the normalization (standardization) constants of a matrix
    """
    return X.mean(axis=0), X.std(axis=0)
