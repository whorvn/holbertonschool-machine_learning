#!/usr/bin/env python3
"""Module for creating mini-batches"""


import tensorflow as tf 


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network in
    tensorflow using
    the Adam optimization algorithm.
    Args:
        alpha: the learning rate.
        beta1: the weight used for the first moment.
        beta2: the weight used for the second moment.
        epsilon: a small number to avoid division by zero.
    Returns:
        The Adam optimization operation.
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
    return optimizer
