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
    init = tf.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)
    Z = layer(prev)
    
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    
    mean, variance = tf.nn.moments(Z, axes=[0])
    epsilon = 1e-8
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, epsilon)
    
    return activation(Z_norm)
