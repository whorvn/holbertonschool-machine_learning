#!/usr/bin/env python3
"""
Transfer Learning with memory optimizations using generators
"""

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np


def resize_and_preprocess(image, size=(96, 96)):
    """Resize and preprocess a single image"""
    resized = cv2.resize(image, size)
    return resized


class DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator to process images on the fly"""
    def __init__(self, x_set, y_set, batch_size=32, img_size=(96, 96), 
                 shuffle=True, augment=False):
        self.x = x_set
        self.y = to_categorical(y_set, num_classes=10)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.datagen = None
        if self.augment:
            self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True
            )
        self.indexes = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get batch data
        batch_x = self.x[indexes]
        batch_y = self.y[indexes]
        
        # Process images
        processed_x = np.array([resize_and_preprocess(img, self.img_size) 
                                for img in batch_x], dtype=np.float32)
        
        # Apply preprocessing
        processed_x = preprocess_input(processed_x)
        
        # Apply augmentation if needed
        if self.augment and self.datagen:
            # Generate augmented data
            for i in range(len(processed_x)):
                if np.random.rand() > 0.5:  # Apply augmentation to 50% of images
                    processed_x[i] = self.datagen.random_transform(processed_x[i])
        
        return processed_x, batch_y

    def on_epoch_end(self):
        """Shuffle indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)


def build_model():
    """
    Builds a lightweight transfer learning model
    """
    # Use a smaller input size
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(96, 96, 3)  # Further reduced size
    )

    # Freeze most layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    # Build model with minimal parameters
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)  # Reduced from 128
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
    """Trains model with memory-efficient generators"""
    # Load CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Convert to flat labels
    Y_train = Y_train.reshape(-1)
    Y_test = Y_test.reshape(-1)
    
    # Create data generators instead of preprocessing all at once
    train_generator = DataGenerator(
        X_train, Y_train, 
        batch_size=16, 
        img_size=(96, 96),
        shuffle=True,
        augment=True
    )
    
    val_generator = DataGenerator(
        X_test, Y_test, 
        batch_size=16, 
        img_size=(96, 96),
        shuffle=False
    )
    
    # Build model
    model = build_model()
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "cifar10_checkpoint.h5",
            save_best_only=True,
            monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            monitor="val_accuracy",
            restore_best_weights=True
        )
    ]
    
    # Train using generators
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,  # Reduced from 20
        callbacks=callbacks,
        workers=1,  # Use just one worker to minimize memory usage
        verbose=1
    )
    
    # Save final model
    model.save("cifar10.h5")


def evaluate_model():
    """Evaluates the model using the generator approach"""
    _, (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
    Y_test = Y_test.reshape(-1)
    
    test_generator = DataGenerator(
        X_test, Y_test,
        batch_size=16,
        img_size=(96, 96),
        shuffle=False
    )
    
    model = tf.keras.models.load_model("cifar10.h5")
    loss, acc = model.evaluate(test_generator, verbose=1)
    print(f"Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    # Use memory growth to avoid TensorFlow taking all GPU memory at once
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
    # Set memory limits
    tf.config.set_logical_device_configuration(
        gpus[0] if gpus else None,
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
    )
    
    # Limit TensorFlow's memory usage
    physical_devices = tf.config.list_physical_devices('CPU')
    if physical_devices:
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
        )
    
    train_model()
    evaluate_model()
