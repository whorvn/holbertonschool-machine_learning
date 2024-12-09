#!/usr/bin/env python3
import numpy as np

"""
This module provides utilities for performing mathematical operations
on matrices using NumPy. It includes functions to multiply matrices
efficiently.

Functions:
- np_matmul(mat1, mat2): Multiplies two matrices and returns the result.
"""

def np_matmul(mat1, mat2):
    """
    Multiplies two matrices (mat1 and mat2) using NumPy's matmul function.

    Parameters:
    - mat1: A NumPy ndarray representing the first matrix.
    - mat2: A NumPy ndarray representing the second matrix.

    Returns:
    - A new NumPy ndarray representing the result of matrix multiplication.
    """
    return np.matmul(mat1, mat2)
