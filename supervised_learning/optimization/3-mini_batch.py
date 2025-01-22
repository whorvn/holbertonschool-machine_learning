#!/usr/bin/env python3
"""
Defines a function that builds a neural network
using Keras library
"""


import numpy as np


def create_mini_batches(X, Y, batch_size):
    """
    creates mini batches from a dataset
    """
    shuffle_data = __import__('2-shuffle_data').shuffle_data
    X, Y = shuffle_data(X, Y)
    mini_batches = []
    shuffle = np.random.permutation(m)
    complete_batches = m // batch_size
    for i in range(complete_batches):
        X_mini = X[i * batch_size:(i + 1) * batch_size]
        Y_mini = Y[i * batch_size:(i + 1) * batch_size]
        mini_batches.append((X_mini, Y_mini))
    if m % batch_size != 0:
        X_mini = X[complete_batches * batch_size:]
        Y_mini = Y[complete_batches * batch_size:]
        mini_batches.append((X_mini, Y_mini))
    return mini_batches
