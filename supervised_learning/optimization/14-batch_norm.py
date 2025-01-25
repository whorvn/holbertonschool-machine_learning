#!/usr/bin/env python3
"""Module for creating mini-batches"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.
    
    Args:
        prev (tf.Tensor): The activated output of the previous layer.
        n (int): The number of nodes in the layer to be created.
        activation (tf.nn.activation): The activation function that should be
        used on the output of the layer.
    
    Returns:
        tf.Tensor: The activated output of the layer.
    """
    # Initialize layer with variance scaling
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, kernel_initializer=init)

    # Get layer output
    z = layer(prev)

    # Calculate mean and variance
    mean, variance = tf.nn.moments(z, axes=[0])

    # Create trainable parameters
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    # Apply batch normalization
    z_norm = tf.nn.batch_normalization(
        z, mean, variance, beta, gamma, epsilon=1e-7)

    # Apply activation function
    return activation(z_norm)
