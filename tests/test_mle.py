from mle import __version__
import mle.classifiers as melc
import pytest
import numpy as np
import pandas as pd
from math import isclose
from sklearn import datasets


def test_version():
    assert __version__ == "0.1.0"


@pytest.fixture
def iris():
    iris = datasets.load_iris()
    names = ["Class"] + iris.feature_names
    targets = iris.target.reshape(len(iris.data), 1)
    df = pd.DataFrame(np.hstack((targets, iris.data)), columns=names)
    df = df[df.Class <= 1]
    df.Class = df.Class.apply(lambda x: 1 if x == 0.0 else -1)
    return df


@pytest.fixture
def params():
    return {"eta": 0.1, "N": 100}


@pytest.fixture
def wx():
    w = np.array([0.2, 0.5, 0.3])
    x = np.array([1, 2, 3])
    return w, x


@pytest.fixture
def wx_n():
    w = np.array([0.2, 0.5, 0.3])
    x = np.array([-1, -2, -3])
    return w, x


@pytest.fixture
def illegal_params():
    return [(-0.1, 10), (1.1, 10)]


class TestClassifier:
    def test_init(self, params):
        c = melc.Classifier(**params)
        assert c.eta, c.maxiter == params

    def test_init_error(self, illegal_params):
        for params in illegal_params:
            with pytest.raises(ValueError):
                melc.Classifier(*params)

    def test_z(self, params, wx):
        c = melc.Classifier(**params)
        assert isclose(c.z(*wx), 2.1)


class TestPerceptron:
    def test_init(self, params):
        p = melc.Perceptron(**params)
        assert p.eta, p.maxiter == params

    def test_z(self, params, wx):
        p = melc.Perceptron(**params)
        assert isclose(p.z(*wx), 2.1)

    def test_phi(self, params, wx, wx_n):
        p = melc.Perceptron(**params)
        assert p.phi(*wx) == 1
        assert p.phi(*wx_n) == -1

    def test_iris(self, iris):
        p = melc.Perceptron(eta=0.01, N=25)
        p.fit(iris.drop("Class", axis=1).to_numpy(), iris.Class.to_numpy())
        assert p.errors[-1] == 0
