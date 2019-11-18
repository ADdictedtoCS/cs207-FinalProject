#Sequence of tests for function.
#TODO-Theo, Junzhi

import pytest
import numpy as np
from autodiff.variable import Variable
import autodiff.function as F

def test_create_function_exception():
    with pytest.raises(NotImplementedError):
        f = F.Function()
        x = Variable(0)
        y = f(x)
    with pytest.raises(NotImplementedError):
        f = F.Function()
        x = Variable(0)
        y = f.get_grad(x)
    with pytest.raises(NotImplementedError):
        f = F.Function()
        x = Variable(0)
        y = f.get_val(x)

def test_exp():
    x = Variable(2)
    exp = F.Exponent()
    y = exp(x)
    assert abs(y.val - np.exp(2)) < 1e-4 and abs(y.grad - np.exp(2)) < 1e-4

def test_sin():
    x = Variable(np.pi / 6)
    sin = F.Sinus()
    y = sin(x)
    assert abs(y.val - 0.5) < 1e-4 and abs(y.grad - np.sqrt(3) / 2) < 1e-4

def test_cos():
    x = Variable(np.pi / 3)
    cos = F.Cosinus()
    y = cos(x)
    assert abs(y.val - 0.5) < 1e-4 and abs(y.grad - np.sqrt(3) / 2) < 1e-4

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
test_sin()
test_cos()
test_exp()
test_tan()
test_tan_exception()
