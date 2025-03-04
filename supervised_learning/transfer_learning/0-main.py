#!/usr/bin/env python3

from tensorflow import keras as K
from tensorflow import keras as K
import tensorflow  as tf
from tensorflow.python.keras import backend
preprocess_data = __import__('0-transfer').resize_and_preprocess

# to fix issue with saving keras applications
K.learning_phase = backend.symbolic_learning_phase()

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)