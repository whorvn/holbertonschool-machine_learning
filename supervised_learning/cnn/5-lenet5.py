#!/usr/bin/env python3
"""documentation documentation
documentation documentation"""


from tensorflow import keras as K


def lenet5(X):
    """Documentation documentation
    Documentation documentation"""
    initializer = K.initializers.HeNormal(seed=0)
    model = K.Sequential()

    model.add(X)

    model.add(K.layers.Conv2D(filters=6,
                              kernel_size=5,
                              padding='same',
                              kernel_initializer=initializer,
                              activation='relu'))

    model.add(K.layers.MaxPooling2D(pool_size=2, strides=2))

    model.add(K.layers.Conv2D(filters=16,
                              kernel_size=5,
                              padding='valid',
                              kernel_initializer=initializer,
                              activation='relu'))

    model.add(K.layers.MaxPooling2D(pool_size=2, strides=2))

    model.add(K.layers.Flatten())

    model.add(K.layers.Dense(units=120,
                             kernel_initializer=initializer,
                             activation='relu'))

    model.add(K.layers.Dense(units=84,
                             kernel_initializer=initializer,
                             activation='relu'))

    model.add(K.layers.Dense(units=10,
                             kernel_initializer=initializer,
                             activation='softmax'))

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
