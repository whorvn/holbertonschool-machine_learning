#!/usr/bin/env python3
"""create a layer for tensorflow"""


import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """create a layer for tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=init, name='layer')
    return layer(prev)
