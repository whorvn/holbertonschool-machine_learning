#!/usr/bin/env python3
"""create a layer for tensorflow"""


import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """create a layer for tensorflow"""
    if activation == tf.nn.tanh:
        return("Tanh")
    elif activation == tf.nn.relu:
        return("Relu")
    elif activation == tf.nn.sigmoid:
        return("Sigmoid")
    init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(n,
                            activation=activation,
                            kernel_initializer=init,
                            name='layer')
    return layer(prev)
