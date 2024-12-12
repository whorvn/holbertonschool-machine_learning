#!/usr/bin/env python3
""" plots y as a line graph """
import numpy as np
import matplotlib.pyplot as plt


def line():

    y = np.arange(0, 11) ** 3
    plt.xlim(0, 10)
    plt.plot(y, color='red')
    plt.show()
