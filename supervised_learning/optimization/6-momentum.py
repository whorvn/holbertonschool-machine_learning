#!/usr/bin/env python3
"""Module for creating mini-batches"""


import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Creates a momentum operation in tensorflow using gradient descent"""
    return tf.train.MomentumOptimizer(alpha, beta1)
