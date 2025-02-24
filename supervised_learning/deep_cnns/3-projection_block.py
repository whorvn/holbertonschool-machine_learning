#!/usr/bin/env python3
"""
Identity Block
"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """documentation documentation
       documentation for projection block"""
    F11, F3, F12 = filters

    # Initializer he_normal with seed 0
    init = K.initializers.HeNormal(seed=0)

    # First layer of left branch (using stride s)
    conv1 = K.layers.Conv2D(filters=F11,
                            kernel_size=(1, 1),
                            strides=(s, s),
                            padding="same",
                            kernel_initializer=init)(A_prev)

    norm1 = K.layers.BatchNormalization(axis=-1)(conv1)
    relu1 = K.layers.Activation(activation="relu")(norm1)

    # Second layer of left branch
    conv2 = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            kernel_initializer=init)(relu1)
    norm2 = K.layers.BatchNormalization(axis=-1)(conv2)
    relu2 = K.layers.Activation(activation="relu")(norm2)

    # Final layer of left branch
    conv3 = K.layers.Conv2D(filters=F12,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding="same",
                            kernel_initializer=init)(relu2)
    norm3 = K.layers.BatchNormalization(axis=-1)(conv3)

    # Right branch: convolve input using F12 with stride s then BatchNorm
    conv_input = K.layers.Conv2D(filters=F12,
                                 kernel_size=(1, 1),
                                 strides=(s, s),
                                 padding="same",
                                 kernel_initializer=init)(A_prev)
    norm_input = K.layers.BatchNormalization(axis=-1)(conv_input)

    # Merge output of left branch and right branch
    merged = K.layers.Add()([norm3, norm_input])

    # Return activated output of merge, using ReLU
    return K.layers.Activation(activation="relu")(merged)
