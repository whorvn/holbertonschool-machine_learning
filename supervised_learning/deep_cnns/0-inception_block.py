#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Documentation Documentation
    Documentation Documentation"""
    F1, F3R, F3,F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=None)
    conv1x1 = K.layers.Conv2D(
        F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
    )(A_prev)
    
    conv3x3_reduce = K.layers.Conv2D(F3R, kernel_size=(1, 1), padding='same',
                                     activation='relu')(conv1x1)
    
    conv3x3 = K.layers.Conv2D(F3, kernel_size=(3, 3), padding='same',
                              activation='relu')(conv3x3_reduce)

    conv5x5_reduce = K.layers.Conv2D(F5R, kernel_size=(1, 1), padding='same',
                                     activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(F5, kernel_size=(5, 5), padding='same',
                              activation='relu')(conv5x5_reduce)

    maxpool = K.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1),
                                 padding='same')(A_prev)
    maxpool_conv = K.layers.Conv2D(FPP, kernel_size=(1, 1), padding='same',
                                   activation='relu')(maxpool)

    output = K.layers.Concatenate(
        axis=-1)([conv1x1, conv3x3, conv5x5, maxpool_conv])

    return output
