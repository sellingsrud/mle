"""
Implementing the following classifiers:
- Perceptron
- Adaline
- Linear Regression
- Logistic Regression
"""

import numpy as np


class Classifier:
    """
    Superclass for classifiers.
    TODO Either force y to have (n, 1) shape or throw error if (n,).
    """

    def __init__(self, eta=0.1, N=100, w=None):
        if eta <= 0 or eta > 1:
            raise ValueError("Learning rate eta must be between 0 and 1")

        self.eta = eta
        self.N = N
        self.w = w
        self.errors = []

    def activation(self, X):
        return X

    def z(self, X):
        """
        Weighted sum of x by w.
        """
        return X @ self.w

    def predict(self, X):
        return np.where(self.activation(self.z(X)) >= 0.0, 1, -1)

    def _add_const(self, X):
        """
        This wastes some memory storing a lot of 1's,
        but make the code below easier to read.
        Call it "pedagogical" to get away with it.
        """
        x0 = np.ones(shape=(X.shape[0], 1))
        return np.hstack((x0, X))

    def _init_fit(self, X):
        nrows, ncols = X.shape

        # Add a constant as the first column
        X = self._add_const(X)

        self._init_w(ncols)

        return X, nrows, ncols

    def _init_w(self, size):
        # Initialize weights if not given
        # Add 1 to size to account for the ones added above
        # TODO generalize
        if self.w is None:
            self.w = np.random.normal(loc=0, scale=0.01, size=(size + 1, 1))


class Perceptron(Classifier):
    """
    The Perceptron classifier.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def activation(self, x):
        return 1 if x > 0 else -1

    def fit(self, X, y):
        # TODO think nrows, ncols are superfluous returns
        X, nrows, ncols = self._init_fit(X)

        for _ in range(self.N):
            errors = 0
            for xi, yi in zip(X, y):
                error = yi - self.predict(xi)
                dw = self.eta * error * xi
                self.w += dw.reshape((len(dw), 1))
                errors += error != 0.0

            self.errors.append(errors)
        return self


class Adaline(Classifier):
    """
    Adaptive Linear Neuron Classifier.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y):
        # TODO think nrows, ncols are superfluous returns
        X, nrows, ncols = self._init_fit(X)

        self.cost_ = list()

        for _ in range(self.N):
            error = y - self.activation(self.z(X))

            # This is just the gradient of J(w) now
            dw = self.eta * error.T @ X
            self.w += dw.T
            cost = 0.5 * (error ** 2).sum()
            self.cost_.append(cost)

        return self


class LinearRegression(Classifier):
    """
    Linear Regression classifier
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_w(self, size):
        self.w = np.zeros((size + 1, 1))

    def _error(self, X, y):
        return y - self.activation(self.z(X))

    def fit(self, X, y):
        """
        Fit model using Linear Regression
        """
        X, nrows, ncols = self._init_fit(X)

        self.mse = []

        for _ in range(self.N):
            error = self._error(X, y)
            dw = self.eta * error.T @ X
            self.w += dw.T
            mse = 0.5 * (error ** 2).sum()
            self.mse.append(mse)

        return self


class LogisticRegression(Classifier):
    def __init__(self, **kwargs):
        raise NotImplementedError
