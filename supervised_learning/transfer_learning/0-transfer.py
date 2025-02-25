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


def create_model():
    """
    Creates a simple convolutional neural network model using Keras
    Returns: model
    model is a Keras model
    """
    base_model = K.applications.ResNet50(include_top=False, weights='imagenet',
                                         input_shape=(224, 224, 3))
    base_model.trainable = False
    model = K.models.Sequential()
    model.add(K.layers.UpSampling2D())
    model.add(base_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(256, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.Dense(10, activation='softmax'))
    return model
