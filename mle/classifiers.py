"""
Implementing the following classifiers:
- Perceptron
"""

import numpy as np


class Classifier:
    """
    Superclass for classifiers.
    """

    def __init__(self, eta=0.1, N=100, w=None, errors=None):
        if eta <= 0 or eta > 1:
            raise ValueError("Learning rate eta must be between 0 and 1")

        self.eta = eta
        self.N = N
        self.w = w
        self.errors = errors

    def z(self, w, x):
        """
        Weighted sum of x by w.
        """
        return w.T @ x


class Perceptron(Classifier):
    """
    The Perceptron classifier.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def phi(self, w, x):
        if self.z(w, x) > 0:
            return 1
        else:
            return -1

    def fit(self, X, y):
        for _ in range(self.N):
            pass
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
