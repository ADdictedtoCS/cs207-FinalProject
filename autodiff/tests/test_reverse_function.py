#Sequence of tests for function.
#TODO-Theo, Junzhi

import pytest
import numpy as np
from autodiff.variable import ReverseVariable
import autodiff.function as F
from autodiff.utils import *

def test_create_function_exception():
    with pytest.raises(NotImplementedError):
        f = F.Function()
        x = ReverseVariable([1, 2, 3])
        y = f(x)
    with pytest.raises(NotImplementedError):
        f = F.Function()
        x = ReverseVariable([1, 2, 3])
        y = f.get_grad(x)
    with pytest.raises(NotImplementedError):
        f = F.Function()
        x = ReverseVariable([1, 2, 3])
        y = f.get_val(x)

def test_exp():
    x = ReverseVariable(2)
    exp = F.Exponent()
    y = exp(x)
    y.reverse()
    assert close(y.val, np.exp(2))
    assert close(x.grad, np.exp(2))

def test_exp_exception():
    x = ReverseVariable([1, 2, 3])
    exp = F.Exponent()
    with pytest.raises(ValueError):
        y = exp(x)

def test_sin():
    x = ReverseVariable(np.pi / 6)
    sin = F.Sinus()
    y = sin(x)
    y.reverse()
    assert abs(y.val - 0.5) < 1e-4 and abs(x.grad - np.sqrt(3) / 2) < 1e-4

def test_cos():
    x = ReverseVariable(np.pi / 3)
    cos = F.Cosinus()
    y = cos(x)
    y.reverse()
    assert abs(y.val - 0.5) < 1e-4 and abs(x.grad + np.sqrt(3) / 2) < 1e-4

def test_tan():
    x = ReverseVariable(np.pi / 4)
    tan = F.Tangent()
    y = tan(x)
    y.reverse()
    assert abs(y.val - 1) < 1e-4 and abs(x.grad - 2) < 1e-4

def test_tan_exception():
    x = ReverseVariable(np.pi / 2)
    tan = F.Tangent()
    with pytest.raises(ValueError):
        y = tan(x)

def test_arcsin():
    x = ReverseVariable(0.5)
    arcsin = F.Arcsin()
    y = arcsin(x)
    y.reverse()
    assert close(y.val, np.pi / 6) and close(x.grad, 1.0 / np.sqrt(1.0 - 0.5 ** 2))

def test_arccos():
    x = ReverseVariable(0.5)
    arccos = F.Arccos()
    y = arccos(x)
    y.reverse()
    assert close(y.val, np.pi / 3) and close(x.grad, -1.0 / np.sqrt(1.0 - 0.5 ** 2))

def test_arctan():
    x = ReverseVariable(1)
    arctan = F.Arctan()
    y = arctan(x)
    y.reverse()
    assert close(y.val, np.pi / 4) and close(x.grad, 0.5)

def test_sinh():
    x = ReverseVariable(1)
    sinh = F.Sinh()
    y = sinh(x)
    y.reverse()
    assert close(y.val, np.sinh(1)) and close(x.grad, np.cosh(1))

def test_cosh():
    x = ReverseVariable(1)
    cosh = F.Cosh()
    y = cosh(x)
    y.reverse()
    assert close(y.val, np.cosh(1)) and close(x.grad, np.sinh(1))

def test_tanh():
    x = ReverseVariable(1)
    tanh = F.Tanh()
    y = tanh(x)
    y.reverse()
    assert close(y.val, np.tanh(1)) and close(x.grad, (np.cosh(1) ** 2 - np.sinh(1) ** 2) / (np.cosh(1) ** 2))

def test_log():
    x = ReverseVariable(2)
    log = F.Log()
    y = log(x)
    y.reverse()
    assert close(y.val, np.log(2)) and close(x.grad, 0.5)
    log2 = F.Log(2)
    y = log2(x)
    y.reverse()
    assert close(y.val, 1) 
    assert close(x.grad, 0.5 / np.log(2))

def test_log_exception():
    with pytest.raises(ValueError):
        log = F.Log(-1)
    with pytest.raises(ValueError):
        log = F.Log([1, 2])

def test_logistic():
    x = ReverseVariable(2)
    logi = F.Logistic()
    y = logi(x)
    y.reverse()
    assert close(y.val, 1.0 / (1.0 + np.exp(-(2)))) and close(x.grad, (1.0 / (1.0 + np.exp(-(2)))) * (1.0 - (1.0 / (1.0 + np.exp(-(2))))))

def test_logistic_exception():
    with pytest.raises(ValueError):
        logi = F.Logistic(L=[1, 2])
    with pytest.raises(ValueError):
        logi = F.Logistic(k=[1, 2])
    with pytest.raises(ValueError):
        logi = F.Logistic(x0=[1, 2])

def test_sqrt():
    x = ReverseVariable(4)
    sqrt = F.Sqrt()
    y = sqrt(x)
    y.reverse()
    assert close(y.val, 2) and close(x.grad, 0.25)

def test_dot():
    x = ReverseVariable([1, 1])
    M = np.asarray(([2, 2], [1, 1]))
    dotm = F.Dot(M)
    y = dotm(x)
    y.reverse()
    assert close(y.val, np.asarray([[4], [2]])) and close(x.grad, M)

test_create_function_exception()
test_exp()
test_exp_exception()
test_sin()
test_cos()
test_exp()
test_tan()
test_tan_exception()
test_arcsin()
test_arccos()
test_arctan()
test_sinh()
test_cosh()
test_tanh()
test_log()
test_logistic()
test_logistic_exception()
test_sqrt()
test_dot()