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
    y = exp(x)
    assert close(y.val, np.exp(2))
    assert close(y.grad, np.exp(2))

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

def test_arcsin():
    x = Variable(0.5)
    arcsin = F.Arcsin()
    y = arcsin(x)
    assert close(y.val, np.pi / 6) and close(y.grad, 1.0 / np.sqrt(1.0 - 0.5 ** 2))

def test_arccos():
    x = Variable(0.5)
    arccos = F.Arccos()
    y = arccos(x)
    assert close(y.val, np.pi / 3) and close(y.grad, -1.0 / np.sqrt(1.0 - 0.5 ** 2))

def test_arctan():
    x = Variable(1)
    arctan = F.Arctan()
    y = arctan(x)
    assert close(y.val, np.pi / 4) and close(y.grad, 0.5)

def test_sinh():
    x = Variable(1)
    sinh = F.Sinh()
    y = sinh(x)
    assert close(y.val, np.sinh(1)) and close(y.grad, np.cosh(1))

def test_cosh():
    x = Variable(1)
    cosh = F.Cosh()
    y = cosh(x)
    assert close(y.val, np.cosh(1)) and close(y.grad, np.sinh(1))

def test_tanh():
    x = Variable(1)
    tanh = F.Tanh()
    y = tanh(x)
    assert close(y.val, np.tanh(1)) and close(y.grad, (np.cosh(1) ** 2 - np.sinh(1) ** 2) / (np.cosh(1) ** 2))

def test_log():
    x = Variable(2)
    log = F.Log()
    y = log(x)
    assert close(y.val, np.log(2)) and close(y.grad, 0.5)
    log2 = F.Log(2)
    y = log2(x)
    assert close(y.val, 1) 
    assert close(y.grad, 0.5 / np.log(2))

def test_log_exception():
    with pytest.raises(ValueError):
        log = F.Log(-1)
    with pytest.raises(ValueError):
        log = F.lLog([1, 2])

def test_logistic():
    x = Variable(2)
    logi = F.Logistic()
    y = logi(x)
    assert close(y.val, 1.0 / (1.0 + np.exp(-(2)))) and close(y.grad, (1.0 / (1.0 + np.exp(-(2)))) * (1.0 - (1.0 / (1.0 + np.exp(-(2))))))

def test_logistic_exception():
    with pytest.raises(ValueError):
        logi = F.Logistic(L=[1, 2])
    with pytest.raises(ValueError):
        logi = F.Logistic(k=[1, 2])
    with pytest.raises(ValueError):
        logi = F.Logistic(x0=[1, 2])

def test_sqrt():
    x = Variable(4)
    sqrt = F.Sqrt()
    y = sqrt(x)
    assert close(y.val, 2) and close(y.grad, -0.5)

def test_dot():
    x = Variable([1, 1])
    M = np.matrix(([2, 2], [1, 1]))
    dotm = F.Dot(M)
    y = dotm(x)
    assert close(y.val, np.matrix([[4], [2]])) and close(y.grad, M)

def test_concat_values_shapes():
    X = Variable([1,2,3])
    x,y,z = X.unroll()
    f1 = x+y 
    f2 = x*y+z
    #=========================
    #Concatenate 2 scalar
    #=========================
    conc = F.concat([f1,f2])
    real_v = np.array([[3, 5]], dtype=np.float64).T
    real_gradients = np.array([[1,1,0], [2,1,1]], dtype=np.float64)
    assert (real_v == conc.val).all(), "Value or Shape Error for the value"
    assert (real_gradients == conc.grad).all(), "Value or Shape Error for the value"
    #), "Value or Shape Error for the value"
    #=========================
    #Concatenate scalar and vector
    #=========================
    new_conc = F.concat([f1,conc])
    real_v = np.array([[3,3, 5]], dtype=np.float64).T
    real_gradients = np.array([ [1, 1, 0], [1, 1, 0], [2, 1, 1] ], dtype=np.float64)
    assert (real_v == new_conc.val).all(), "Value or Shape Error for the value"
    assert (real_gradients == new_conc.grad).all(
    ), "Value or Shape Error for the value"
    #=========================
    #Concatenate vector and vector
    #=========================
    full_conc = F.concat([new_conc, conc])
    real_v = np.array([[3, 3, 5, 3,5]], dtype=np.float64).T
    real_gradients = np.array([[1, 1, 0], [1, 1, 0], [2, 1, 1], 
    [1, 1, 0], [2, 1, 1]], dtype=np.float64)
    assert (real_v == full_conc.val).all(), "Value or Shape Error for the value"
    assert (real_gradients == full_conc.grad).all(), "Value or Shape Error for the value"


def test_concat_exception():
    X = Variable([1, 2, 3])
    Y = Variable([1,2])
    _, _, x = X.unroll()
    _, y = Y.unroll()
    with pytest.raises(AssertionError):
        f = F.concat([x,y])
    with pytest.raises(AssertionError):
        f = F.concat([])








    
    


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
test_concat_values_shapes()
test_concat_exception()
