#!/usr/bin/env python3
"""
documentation documentation
documentation documentation
documentation documentation
"""


import numpy as np


class Neuron:
    """
    documentation documentation
    documentation documentation
    documentation documentation
    """
    def __init__(self, nx):
        """
        documentation documentation
        documentation documentation
        documentation documentation
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
        self.nx = nx
