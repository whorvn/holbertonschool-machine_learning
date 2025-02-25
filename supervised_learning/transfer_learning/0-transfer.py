#!/usr/bin/env python3
"""
Transfer Learning
"""


from tensorflow import keras as K
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np


def preprocess_data(X, Y):
    """
    Pre-processes the data for your model:
    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
    where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
    X_p is a numpy.ndarray containing the preprocessed X
    Y_p is a numpy.ndarray containing the preprocessed Y
    """
    """Preprocess CIFAR-10 images and labels for EfficientNetB0."""
    X_resized = np.array([cv2.resize(img, (224, 224)) for img in X])  # Resize images
    X_scaled = preprocess_input(X_resized)  # Normalize using EfficientNet preprocessing
    Y_categorical = to_categorical(Y, num_classes=10)  # Convert labels to one-hot encoding
    return X_scaled, Y_categorical


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def build_model():
    """
    Builds and returns a transfer learning model using EfficientNetB0 for
    CIFAR-10 classification.
    """
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers[:-20]:  # Freeze most layers except the last few
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Reduce feature maps to a vector
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)  # Prevent overfitting
    x = Dense(10, activation="softmax")(x)  # Output layer for 10 classes

    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

    return model
