#!/usr/bin/env python3
"""Module for creating mini-batches"""


import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow.

    Args:
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight (discounting factor).
        epsilon (float): A small number to avoid division by zero.

    Returns:
        optimizer: A TensorFlow RMSProp optimizer configured with the given parameters.
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
    return optimizer
