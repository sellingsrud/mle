"""
Implementing the following classifiers:
- Perceptron
"""

import numpy as np


class Classifier:
    """
    Superclass for classifiers.
    """

    def __init__(self, eta=0.1, N=100, w=None):
        if eta <= 0 or eta > 1:
            raise ValueError("Learning rate eta must be between 0 and 1")

        self.eta = eta
        self.N = N
        self.w = w
        self.errors = []

    def z(self, x):
        """
        Weighted sum of x by w.
        """
        return self.w.T @ x


class Perceptron(Classifier):
    """
    The Perceptron classifier.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def phi(self, x):
        return 1 if self.z(x) > 0 else -1

    def fit(self, X, y):
        nrows, ncols = X.shape

        # Add a constant as the first column
        x0 = np.ones(shape=(nrows, 1))
        X = np.hstack((x0, X))

        # Initialize weights if not given
        # Add 1 to size to account for the ones added above
        if self.w is None:
            self.w = np.random.normal(loc=0, scale=0.01, size=ncols + 1)

        for _ in range(self.N):
            error = y - np.array([self.phi(x) for x in X])
            dw = self.eta * error @ X
            self.w += dw.flatten()
            self.errors.append(np.where(error != 0, 1, 0).sum())

        return self

    def predict(self, X):
        return np.where(self.z(X) >= 0.0, 1, -1)
