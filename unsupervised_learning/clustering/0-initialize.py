#!/usr/bin/env python3

"""
Initialize cluster centroids
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means clustering.

    Parameters:
    X (numpy.ndarray): A 2D numpy array of shape (n, d) containing the dataset
                       that will be used for K-means clustering.
                       - n is the number of data points
                       - d is the number of dimensions for each data point
    k (int): A positive integer representing the number of clusters.

    Returns:
    numpy.ndarray: A 2D numpy array of shape (k, d) containing the initialized
                   centroids for each cluster.
                   Returns None on failure
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(k) is not int or k <= 0:
        return None

    n, d = X.shape
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return np.random.uniform(min_val, max_val, (k, d))
