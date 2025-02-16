#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """documentation documentation
    documentation documentation"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    conv2d_1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(x)

    max_pooling_1 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(conv2d_1)

    conv2d_2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(max_pooling_1)

    max_pooling_2 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(conv2d_2)

    # Flatten tensor to 1D tensor, to match Dense layer dimensions
    flattened = tf.layers.Flatten()(max_pooling_2)

    fc1 = tf.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(flattened)

    fc2 = tf.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(fc1)

    output = tf.layers.Dense(
        units=10,
        kernel_initializer=init
    )(fc2)

    softmax = tf.nn.softmax(output)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    y_pred = tf.argmax(output, axis=1)
    y_true = tf.argmax(y, axis=1)
    correct_prediction = tf.equal(y_pred, y_true)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

    return softmax, train_op, loss, accuracy
