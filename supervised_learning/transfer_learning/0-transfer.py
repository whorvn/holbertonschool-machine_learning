#!/usr/bin/env python3
"""
Transfer Learning with memory optimizations
"""

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np


def preprocess_data(X, Y):
    """
    Pre-processes the data in batches to avoid memory issues
    """
    # Process in smaller batches to save memory
    batch_size = 1000
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size + (1 if n_samples % batch_size != 0 else 0)
    
    # Initialize output arrays
    X_p = np.zeros((n_samples, 160, 160, 3))  # Reduced size from 224x224
    Y_p = to_categorical(Y, num_classes=10)
    
    # Process each batch
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        
        # Process current batch
        batch = X[start_idx:end_idx]
        X_p[start_idx:end_idx] = np.array([cv2.resize(img, (160, 160)) for img in batch])
    
    # Apply preprocessing
    X_p = preprocess_input(X_p)
    return X_p, Y_p


def build_model():
    """
    Builds a more memory-efficient transfer learning model
    """
    # Use a smaller input size
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(160, 160, 3)  # Reduced from 224x224
    )

    # Freeze most layers to reduce training parameters
    for layer in base_model.layers[:-15]:
        layer.trainable = False

    # Build model with fewer parameters in the classifier
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)  # Reduced from 256
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)
    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def train_model():
    """Trains model with memory optimizations."""
    # Load CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Convert labels to more memory-efficient format
    Y_train = Y_train.reshape(-1)
    Y_test = Y_test.reshape(-1)
    
    # Memory-efficient preprocessing
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)
    
    # Use memory-efficient data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    # Build model
    model = build_model()
    
    # Use callbacks for checkpointing to recover from failures
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "cifar10_checkpoint.h5",
            save_best_only=True,
            monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            monitor="val_accuracy"
        )
    ]
    
    # Train with smaller batch size
    model.fit(
        datagen.flow(X_train, Y_train, batch_size=16),
        validation_data=(X_test, Y_test),
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save("cifar10.h5")


def evaluate_model():
    """Evaluates the trained model on CIFAR-10 test data."""
    _, (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
    X_test, Y_test = preprocess_data(X_test, Y_test)

    model = tf.keras.models.load_model("cifar10.h5")
    loss, acc = model.evaluate(X_test, Y_test, verbose=1)
    print(f"Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    # Use memory growth to avoid TensorFlow taking all GPU memory at once
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    train_model()
    evaluate_model()
