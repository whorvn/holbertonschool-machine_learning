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
    X_p = preprocess_input(X_resized)  # Normalize using EfficientNet preprocessing
    Y_p = to_categorical(Y, num_classes=10)  # Convert labels to one-hot encoding
    return X_p, Y_p


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


def build_model():
    """
    Builds and returns a transfer learning model using EfficientNetB0 for
    CIFAR-10 classification.
    """
    base_model = EfficientNetB0(weights="imagenet",
                                include_top=False,
                                input_shape=(224, 224, 3))

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


from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_model():
    """Trains and saves the CIFAR-10 classifier model."""
    # Load CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # Build and train model
    model = build_model()
    history = model.fit(datagen.flow(X_train, Y_train, batch_size=32),
                        validation_data=(X_test, Y_test),
                        epochs=25,
                        verbose=1)

    # Save model
    model.save("cifar10.h5")

if __name__ == "__main__":
    train_model()


def evaluate_model():
    """Evaluates the trained model on CIFAR-10 test data."""
    _, (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
    X_test, Y_test = preprocess_data(X_test, Y_test)

    model = tf.keras.models.load_model("cifar10.h5")
    loss, acc = model.evaluate(X_test, Y_test, verbose=1)
    print(f"Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model()
