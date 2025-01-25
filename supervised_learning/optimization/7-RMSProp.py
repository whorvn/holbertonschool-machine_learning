#!/usr/bin/env python3
"""Module for creating mini-batches"""


import tensorflow as tf


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates a variable using the RMSProp optimization algorithm.
    Args:
        alpha: the learning rate.
        beta2: the RMSProp weight.
        epsilon: a small number to avoid division by zero.
        var: numpy.ndarray containing the variable to be updated.
        grad: numpy.ndarray containing the gradient of var.
        s: the previous second moment of var.
    Returns:
        The updated variable and the new moment, respectively.
    """
    s = beta2 * s + (1 - beta2) * grad**2
    var = var - alpha * grad / (s**0.5 + epsilon)
    return var, s
