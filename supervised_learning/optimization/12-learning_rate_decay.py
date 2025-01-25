#!/usr/bin/env python3
"""Module for creating mini-batches"""


import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation using
    inverse time decay.

    Args:
        alpha (float): The original learning rate.
        decay_rate (float): The weight used to determine
        the rate at which alpha will decay.
        decay_step (int): The number of passes of gradient
        descent before alpha is decayed further.

    Returns:
        tf.Tensor: The learning rate decay operation.
    """
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True  # Ensures stepwise decay
    )
    return lr_schedule
