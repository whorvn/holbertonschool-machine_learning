#!/usr/bin/env python3
"""
Defines a function that builds a neural network
using Keras library
"""


import numpy as np


def normalize(X, m, s):
    """
    normalizes (standardizes) a matrix
    """
    return (X - m) / s
