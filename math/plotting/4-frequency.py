#!/usr/bin/env python3
"""Document"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Document"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.hist(student_grades,  bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], edgecolor='black')
    plt.xlabel('Project A')
    plt.ylabel('Number of Students')
    plt.title('Grades')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.show()

