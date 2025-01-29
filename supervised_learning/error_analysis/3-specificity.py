supervised_learning/error_analysis/2-precision.py#!/usr/bin/env python3
"""
documentation documentation documentation
"""


import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a
    confusion matrix
    Args:
        confusion: numpy.ndarray of shape (classes, classes)
                   where row indices represent the
                   correct labels
                   and column indices represent the
                   predicted labels
    Returns:
        numpy.ndarray of shape (classes,) containing
        the specificity
        of each class
    """
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives
    true_negatives = np.sum(confusion) - (true_positives +
                                          false_positives +
                                          false_negatives)
    return true_negatives / (true_negatives + false_positives)
