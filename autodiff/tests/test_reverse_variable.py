#Sequence of tests for the variables operations.

import pytest
import numpy as np
from autodiff.variable import ReverseVariable
from autodiff.utils import *

def test_create_variable():
    X = ReverseVariable([1, 2, 3])
    assert close(X.val, np.asarray([[1], [2], [3]]))
    y = ReverseVariable(1)
    assert close(y.val, 1)

def test_add():
    x = ReverseVariable([1, 2])
    y = ReverseVariable([2, 1])
    z = x + y
    z.reverse()
    assert close(z.val, np.asarray([[3], [3]]))
    assert close(x.grad, np.eye(2))
    assert close(y.grad, np.eye(2))
    a = z + [4.4, 3.3]
    b = a + z
    b.reverse()
    assert close(b.val, np.asarray([[10.4], [9.3]]))
    assert close(x.grad, np.eye(2) * 2)
    assert close(y.grad, np.eye(2) * 2)
    c = [4.4, 3.3] + b
    c.reverse()
    assert close(c.val, np.asarray([[14.8], [12.6]]))
    assert close(x.grad, np.eye(2) * 2)
    assert close(y.grad, np.eye(2) * 2)
    
def test_add_exception():
    x = ReverseVariable([1, 2, 3])
    with pytest.raises(TypeError):
        y = x + "g"
    with pytest.raises(TypeError):
        y = True + x

def test_sub():
    x = ReverseVariable([1, 2])
    y = ReverseVariable([2, 1])
    z = y + x
    a = z + [4.4, 3.3]
    b = a + z
    c = b - x
    c.reverse()
    assert close(c.val, np.asarray([[9.4], [7.3]]))
    assert close(x.grad, np.eye(2))
    assert close(y.grad, np.eye(2) * 2)
    d = 3 - c
    d.reverse()
    assert close(d.val, np.asarray([[-6.4], [-4.3]]))
    assert close(x.grad, np.eye(2) * -1)
    assert close(y.grad, np.eye(2) * -2)
    e = c - 2.1
    e.reverse()
    assert close(e.val, np.asarray([[7.3], [5.2]]))
    assert close(x.grad, np.eye(2))
    assert close(y.grad, np.eye(2) * 2)

def test_sub_exception():
    x = ReverseVariable([1, 2])
    y = ReverseVariable([2, 1])
    z = y + x
    a = z + [4.4, 3.3]
    b = a + z
    with pytest.raises(TypeError):
        c = b - "g"
    with pytest.raises(TypeError):
        c = True - b

def test_mul():
    x = ReverseVariable([1, 2])
    y = ReverseVariable([2, 1])
    z = ReverseVariable(4)
    a = x + y
    b = a * z
    b.reverse()
    assert close(b.val, np.asarray([[12], [12]]))
    assert close(x.grad, np.eye(2) * 4)
    assert close(y.grad, np.eye(2) * 4)
    assert close(z.grad, np.asarray([[3], [3]]))
    c = b * 4
    c.reverse()
    assert close(c.val, np.asarray([[48], [48]]))
    assert close(x.grad, np.eye(2) * 16)
    assert close(y.grad, np.eye(2) * 16)
    assert close(z.grad, np.asarray([[12], [12]]))
    d = 2 * b
    d.reverse()
    assert close(d.val, np.asarray([[24], [24]]))
    assert close(x.grad, np.eye(2) * 8)
    assert close(y.grad, np.eye(2) * 8)
    assert close(z.grad, np.asarray([[6], [6]]))
    
def test_mul_exception():
    x = ReverseVariable([1, 2])
    y = ReverseVariable([2, 1])
    z = ReverseVariable(4)
    a = x + y
    with pytest.raises(TypeError):
        b = a * "g"
    with pytest.raises(TypeError):
        b = False * y    

def test_div():
    x = ReverseVariable([1, 3])
    y = ReverseVariable([3, 1])
    z = ReverseVariable(4)
    a = x + y
    b = a / z
    b.reverse()
    assert close(b.val, np.asarray([[1], [1]]))
    assert close(x.grad, np.eye(2) * 0.25)
    assert close(y.grad, np.eye(2) * 0.25)
    assert close(z.grad, np.asarray([[-0.25], [-0.25]]))
    c = b / 0.25
    c.reverse()
    assert close(c.val, np.asarray([[4], [4]]))
    assert close(x.grad, np.eye(2))
    assert close(y.grad, np.eye(2))
    assert close(z.grad, np.asarray([[-1], [-1]]))
    d = 4 / z
    d.reverse()
    assert close(d.val, 1)
    assert close(z.grad, -0.25)

def test_div_exception():
    x = ReverseVariable([1, 3])
    y = ReverseVariable([3, 1])
    z = ReverseVariable(0)
    a = x + y
    with pytest.raises(ValueError):
        b = a / x
    with pytest.raises(ValueError):
        b = a / z
    with pytest.raises(ValueError):
        b = 4.0 / z
    with pytest.raises(ValueError):
        b = a / 0.0
    with pytest.raises(ValueError):
        b = a / np.asarray([1, 1])
    with pytest.raises(ValueError):
        b = 4.0 / x
    with pytest.raises(TypeError):
        b = a / "g"
    with pytest.raises(TypeError):
        a = True / y

def test_pow():
    x = ReverseVariable([1, 3])
    y = ReverseVariable([3, 1])
    z = ReverseVariable(2)
    a = x + y
    b = z ** 2
    b.reverse()
    assert close(b.val, 4)
    assert close(z.grad, 4)
    b = a ** 2
    b.reverse()
    assert close(b.val, np.asarray([[16], [16]]))
    assert close(x.grad, np.eye(2) * 8)
    assert close(y.grad, np.eye(2) * 8)

def test_pow_exception():
    x = ReverseVariable([1, 3])
    y = ReverseVariable([3, 1])
    z = ReverseVariable(2)
    a = x + y
    with pytest.raises(TypeError):
        b = a ** "g"

def test_rpow():
    z = ReverseVariable(2)
    b = 2 ** z
    b.reverse()
    assert close(b.val, 4)
    assert close(z.grad, np.log(2) * 4)

def test_rpow_exception():
    x = ReverseVariable([1, 3])
    y = ReverseVariable([3, 1])
    z = ReverseVariable(2)
    a = x + y
    with pytest.raises(TypeError):
        b = "g" ** y
    with pytest.raises(ValueError):
        b = 2 ** a

def test_neg():
    x = ReverseVariable([1, 3])
    y = ReverseVariable([3, 1])
    z = ReverseVariable(4)
    a = x + y
    b = a / z
    c = -b
    c.reverse()
    assert close(c.val, -np.asarray([[1], [1]]))
    assert close(x.grad, np.eye(2) * -0.25)
    assert close(y.grad, np.eye(2) * -0.25)
    assert close(z.grad, np.asarray([[0.25], [0.25]]))

test_create_variable()
test_add()
test_add_exception()
test_sub()
test_sub_exception()
test_mul()
test_mul_exception()
test_div()
test_div_exception()
test_pow()
test_pow_exception()
test_rpow()
test_rpow_exception()
test_neg()