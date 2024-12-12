#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots a cubic curve using matplotlib and numpy.

    The function generates values from 0 to 10, raises them to the power of 3,
    and then plots the resulting curve using matplotlib. The x-axis is limited
    to the range 0-10, and the plot is displayed in red.
    """
    y = np.arange(0, 11) ** 3
    plt.xlim(0, 10)
    plt.plot(y, color='red')
    plt.show()
