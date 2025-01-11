#!/usr/bin/env python3
"""Defining placeholders for the network"""


import tensorflow as tf
import numpy as np


def create_placeholders(nx, classes):
    """Create placeholders for the network"""
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
