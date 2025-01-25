#!/usr/bin/env python3
"""Module for creating mini-batches"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
"""
    Creates a batch normalization layer for a
    neural network in TensorFlow.

    Args:
        prev (tensor): The activated output of the previous
        layer.
        n (int): The number of nodes in the layer to be
        created.
        activation (function): The activation function
        to use on the output of the layer.

    Returns:
        tensor: A tensor of the activated output for the
        layer.
    """
    # Initialize the Dense layer with the specified kernel initializer
    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'),
        use_bias=False  # Bias is not needed because batch normalization will handle it
    )
    
    # Apply the Dense layer to the input
    Z = dense_layer(prev)
    
    # Initialize gamma and beta as trainable parameters
    gamma = tf.Variable(tf.ones([n]), trainable=True, name="gamma")
    beta = tf.Variable(tf.zeros([n]), trainable=True, name="beta")
    
    # Calculate mean and variance of the batch
    mean, variance = tf.nn.moments(Z, axes=[0])
    
    # Apply batch normalization
    epsilon = 1e-7
    Z_norm = tf.nn.batch_normalization(
        Z,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon
    )
    
    # Apply the activation function
    A = activation(Z_norm)
    
    return A