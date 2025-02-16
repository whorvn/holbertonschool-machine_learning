#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


import tensorflow.compat.v1 as tf


def lenet5(X, Y):
    """documentation documentation
    documentation documentation"""
    init = tf.initializers.VarianceScaling(scale=2.0)
    activation = tf.nn.relu
    conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                                   activation=activation, kernel_initializer=init)(X)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                                   activation=activation, kernel_initializer=init)(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    flatten = tf.keras.layers.Flatten()(pool2)
    fc1 = tf.keras.layers.Dense(units=120, activation=activation,
                                kernel_initializer=init)(flatten)
    fc2 = tf.keras.layers.Dense(units=84, activation=activation,
                                kernel_initializer=init)(fc1)
    output = tf.keras.layers.Dense(units=10, kernel_initializer=init)(fc2)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output))
    train_op = tf.compat.v1.train.AdamOptimizer().minimize(loss)
    y_pred = tf.nn.softmax(output)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return y_pred, train_op, {'accuracy': accuracy}
