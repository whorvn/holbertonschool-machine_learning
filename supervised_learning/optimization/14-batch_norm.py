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
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        n,
        activation=activation,
        name="layer",
        kernal_initializer=weights_initializer)
    x = layer[prev]
    gamma = tf.Variable(tf.constant(
        1, shape=(1, n), trainable=True, name="gamma"))
    beta = tf.Variable(tf.constant(
        0, shape=(1, n), trainable=True, name="gamma"))
    Z = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-8)
    return Z
