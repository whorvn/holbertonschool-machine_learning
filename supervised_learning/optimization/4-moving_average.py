#!/usr/bin/env python3
"""Module for creating mini-batches"""


import numpy as np


def moving_average(data, beta):
    """
    Calculates weighted moving average of a data set
    Args:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average
    Returns:
        list containing the moving averages of data
    """
    moving_averages = []
    prev_avg = 0

    for i in range(len(data)):
        moving_avg = beta * prev_avg + (1 - beta) * data[i]

        bias_correction = 1 - beta ** (i + 1)
        corrected_avg = moving_avg / bias_correction

        moving_averages.append(corrected_avg)
        prev_avg = moving_avg

    return moving_averages
    