#!/usr/bin/env python3
"""
Inception Network
"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    'Deep Residual Learning for Image Recognition' (2015).

    This function constructs an identity block as described in
    the paper. All convolutions inside the block should be
    followed by batch normalization along the channels axis and
    a rectified linear activation (ReLU), respectively.

    Arguments:
    A_prev : tf.Tensor
        The output from the previous layer.
    filters : list
        A list containing the number of filters in each layer
        of the identity block.

    Returns:
    tf.Tensor
        The activated output of the identity block.
    """
    F11, F3, F12 = filters

    # 1x1 conv. layer
    conv1 = K.layers.Conv2D(filters=F11,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding="same",
                            kernel_initializer="he_normal")(A_prev)
    batch_norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    activation1 = K.layers.Activation("relu")(batch_norm1)

    # 3x3 conv. layer
    conv2 = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            kernel_initializer="he_normal")(activation1)
    batch_norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    activation2 = K.layers.Activation("relu")(batch_norm2)

    # 1x1 conv. layer
    conv3 = K.layers.Conv2D(filters=F12,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding="same",
                            kernel_initializer="he_normal")(activation2)
    batch_norm3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Add input tensor to output tensor
    add = K.layers.Add()([batch_norm3, A_prev])

    # Activation
    output = K.layers.Activation("relu")(add)

    return output
