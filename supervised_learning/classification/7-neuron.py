#!/usr/bin/env python3
"""
documentation documentation
documentation documentation
documentation documentation
"""

import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    """
    documentation documentation
    documentation documentation
    documentation documentation
    """
    def __init__(self, nx):
        """
        documentation documentation
        documentation documentation
        documentation documentation
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
        self.nx = nx

    @property
    def W(self):
        """
        gets the private instance attribute __W
        __W is the weights vector for the neuron
        """
        return (self.__W)

    @property
    def b(self):
        """
        gets the private instance attribute __b
        __b is the bias for the neuron
        """
        return (self.__b)

    @property
    def A(self):
        """
        gets the private instance attribute __A
        __A is the activated output of the neuron
        """
        return (self.__A)

    def forward_prop(self, X):
        """
        documentation documentation
        documentation documentation
        documentation documentation
        """
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + (np.exp(-z)))
        return (self.A)

    def cost(self, Y, A):
        """
        documentation documentation
        documentation documentation
        documentation documentation
        """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return (cost)

    def evaluate(self, X, Y):
        """
        documentation documentation
        documentation documentation
        documentation documentation
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        documentation documentation
        documentation documentation
        documentation documentation
        """
        m = Y.shape[1]
        dz = A - Y
        db = np.sum(dz) / m
        dw = np.matmul(X, dz.T) / m
        self.__W = self.W - (alpha * dw).T
        self.__b = self.b - alpha * db
        return (self.W, self.b)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        documentation documentation
        documentation documentation
        documentation documentation
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a number")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        all_cost = []
        all_iterations = []
        for i in range(iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            cost = self.cost(Y, A)
            if verbose is True:
                if i % step == 0:
                    print("Cost after {} iterations: {}".format(i, cost))
                    all_cost.append(cost)
                    all_iterations.append(i)
        if graph is True:
            plt.plot(all_iterations, all_cost, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return (self.evaluate(X, Y))
