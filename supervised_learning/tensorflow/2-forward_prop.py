#!/usr/bin/env python3
"""create a layer for tensorflow"""


import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """forward propagation of a neural network"""
    create_layer = __import__('1-create_layer').create_layer
    for i in range(len(layer_sizes)):
        if i == 0:
            layer = create_layer(x, layer_sizes[i], activations[i])
        else:
            layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer
