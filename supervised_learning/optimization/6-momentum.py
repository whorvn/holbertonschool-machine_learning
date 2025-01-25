#!/usr/bin/env python3
"""Module for creating mini-batches"""


import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm in TensorFlow.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.

    Returns:
        optimizer: A TensorFlow optimizer instance.
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
