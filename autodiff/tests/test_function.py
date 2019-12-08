#Sequence of tests for function.
#TODO-Theo, Junzhi

import pytest
import numpy as np
from autodiff.variable import Variable
import autodiff.function as F

def close(x, y, tol=1e-5):
    if isinstance(x, float):
        return np.abs(x - y) < tol
    if x.shape != y.shape:
        return False
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if np.abs(x[i, j] - y[i, j]) > tol:
                return False
    return True

def test_create_function_exception():
    with pytest.raises(NotImplementedError):
        f = F.Function()
        x = Variable([1, 2, 3])
        y = f(x)
    with pytest.raises(NotImplementedError):
        f = F.Function()
        x = Variable([1, 2, 3])
        y = f.get_grad(x)
    with pytest.raises(NotImplementedError):
        f = F.Function()
        x = Variable([1, 2, 3])
        y = f.get_val(x)

def test_exp():
    x = Variable(2)
    exp = F.Exponent()
    print("hi")
    y = exp(x)
    assert close(y.val, np.exp(2))
    assert close(y.grad[x], np.exp(2))

def test_exp_exception():
    x = Variable([1, 2, 3])
    exp = F.Exponent()
    with pytest.raises(ValueError):
        y = exp(x)

def test_sin():
    x = Variable(np.pi / 6)
    sin = F.Sinus()
    y = sin(x)
    assert abs(y.val - 0.5) < 1e-4 and abs(y.grad - np.sqrt(3) / 2) < 1e-4

def test_cos():
    x = Variable(np.pi / 3)
    cos = F.Cosinus()
    y = cos(x)
    assert abs(y.val - 0.5) < 1e-4 and abs(y.grad + np.sqrt(3) / 2) < 1e-4

def test_tan():
    x = Variable(np.pi / 4)
    tan = F.Tangent()
    y = tan(x)
    assert abs(y.val - 1) < 1e-4 and abs(y.grad - 2) < 1e-4

def test_tan_exception():
    x = Variable(np.pi / 2)
    tan = F.Tangent()
    with pytest.raises(ValueError):
        y = tan(x)

test_create_function_exception()
test_exp()
test_exp_exception()
# test_sin()
# test_cos()
# test_exp()
# test_tan()
# test_tan_exception()
