from mle import __version__
import mle.classifiers as melc
import pytest
import numpy as np
from math import isclose


def test_version():
    assert __version__ == "0.1.0"


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
