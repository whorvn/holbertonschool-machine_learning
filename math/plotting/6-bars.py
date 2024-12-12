#!/usr/bin/env python3
"""docstring for"""
import numpy as np
import matplotlib.pyplot as plt

def bars():
    """docstring for"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    names = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    for i in range(len(fruit)):
        plt.bar(names, fruit[i], bottom=np.sum(fruit[:i], axis=0), color=colors[i], label=fruits[i])
        
    plt.ylabel('Quantity of Fruit')
    plt.yticks(np.arange(0, 90, 10))
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.show()