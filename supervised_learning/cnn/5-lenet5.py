#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


from tensorflow import keras as K


def lenet5(x, y):
    """Documentation documentation
    Documentation documentation"""
    init = K.initializers.VarianceScaling(scale=2.0)
    conv2d_1 = K.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(x)

    max_pooling_1 = K.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(conv2d_1)

    conv2d_2 = K.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(max_pooling_1)

    max_pooling_2 = K.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(conv2d_2)

    # Flatten tensor to 1D tensor, to match Dense layer dimensions
    flattened = K.layers.Flatten()(max_pooling_2)

    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(flattened)

    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(fc1)

    output = K.layers.Dense(
        units=10,
        kernel_initializer=init
    )(fc2)

    softmax = K.activations.softmax(output)
    loss = K.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
    return softmax, loss
