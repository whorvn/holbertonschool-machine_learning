#!/usr/bin/env python3
"""Module for creating mini-batches"""


import numpy as np


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches from dataset
    Args:
        X: numpy.ndarray of shape (m, nx)
        Y: numpy.ndarray of shape (m, ny)
        batch_size: size of each mini-batch
    Returns:
        list of tuples (X_mini, Y_mini)
    """
    m = X.shape[0]
    
    # Input validation
    if not isinstance(batch_size, int) or batch_size <= 0:
        return None
    
    # Shuffle data
    shuffle_data = __import__('2-shuffle_data').shuffle_data
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    
    # Initialize mini-batches list
    mini_batches = []
    
    # Calculate complete batches
    n_complete = m // batch_size
    
    # Create complete mini-batches
    for i in range(n_complete):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        mini_batches.append((
            X_shuffled[start_idx:end_idx],
            Y_shuffled[start_idx:end_idx]
        ))
    
    # Handle remaining samples
    if m % batch_size != 0:
        start_idx = n_complete * batch_size
        mini_batches.append((
            X_shuffled[start_idx:],
            Y_shuffled[start_idx:]
        ))
    
    return mini_batches
