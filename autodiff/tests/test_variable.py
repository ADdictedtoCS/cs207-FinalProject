#Sequence of tests for the variables operations.

import pytest
import numpy as np
from autodiff.variable import Variable
from autodiff.utils import *

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

def test_create_variable_exception():
    with pytest.raises(TypeError):
        x = Variable(True)
    with pytest.raises(TypeError):
        x = Variable("haha")
    with pytest.raises(TypeError):
        x = Variable(2.2, True)
    with pytest.raises(TypeError):
        x = Variable(2.2, "haha")

def test_add():
    x = Variable([1, 2, 3])
    y = Variable([3, 2, 1])
    z = x + y
    assert close(z.val, np.matrix([[4], [4], [4]]))
    assert close(z.grad[x], np.eye(3))
    assert close(z.grad[y], np.eye(3))
    a = z + 4.4
    b = a + z
    assert close(b.val, np.matrix([[12.4], [12.4], [12.4]]))
    assert close(b.grad[x], np.eye(3) * 2)
    assert close(b.grad[y], np.eye(3) * 2)
    
def test_add_exception():
    x = Variable([1, 2, 3])
    with pytest.raises(TypeError):
        y = x + "g"
    with pytest.raises(TypeError):
        y = True + x

def test_sub():
    x = Variable([1, 2, 3])
    y = Variable([3, 2, 1])
    z = y + x
    a = z + 4.4
    b = a + z
    c = b - x
    assert close(c.val, np.matrix([[11.4], [10.4], [9.4]]))
    assert close(c.grad[x], np.eye(3))
    assert close(c.grad[y], np.eye(3) * 2)
    d = 3 - c
    assert close(d.val, np.matrix([[-8.4], [-7.4], [-6.4]]))
    assert close(d.grad[x], -np.eye(3))
    assert close(d.grad[y], -np.eye(3) * 2)
    e = c - 2.1
    assert close(e.val, np.matrix([[9.3], [8.3], [7.3]]))
    assert close(e.grad[x], np.eye(3))
    assert close(e.grad[y], np.eye(3) * 2)

def test_sub_exception():
    x = Variable([1, 2, 3])
    y = Variable([3, 2, 1])
    z = y + x
    a = z + 4.4
    b = a + z
    with pytest.raises(TypeError):
        c = b - "g"
    with pytest.raises(TypeError):
        c = True - b

def test_mul():
    x = Variable([1, 2, 3])
    y = Variable([3, 2, 1])
    z = Variable(4)
    a = x + y
    b = a * z
    assert close(b.val, np.matrix([[16], [16], [16]]))
    assert close(b.grad[x], np.eye(3) * 4)
    assert close(b.grad[y], np.eye(3) * 4)
    assert close(b.grad[z], np.matrix([[4], [4], [4]]))
    c = b * 4
    assert close(c.val, np.matrix([[64], [64], [64]]))
    assert close(c.grad[x], np.eye(3) * 16)
    assert close(c.grad[y], np.eye(3) * 16)
    assert close(c.grad[z], np.matrix([[16], [16], [16]]))
    d = 2 * b
    assert close(d.val, np.matrix([[32], [32], [32]]))
    assert close(d.grad[x], np.eye(3) * 8)
    assert close(d.grad[y], np.eye(3) * 8)
    assert close(d.grad[z], np.matrix([[8], [8], [8]]))
    M = np.eye(3) * 3
    e = b.__rmul__(M)
    assert close(e.val, np.matrix([[48], [48], [48]]))
    assert close(e.grad[x], np.eye(3) * 12)
    assert close(e.grad[y], np.eye(3) * 12)
    assert close(e.grad[z], np.matrix([[12], [12], [12]]))
    
def test_mul_exception():
    x = Variable([1, 2, 3])
    y = Variable([3, 2, 1])
    z = Variable(4)
    a = x + y
    with pytest.raises(TypeError):
        b = a * "g"
    with pytest.raises(TypeError):
        b = False * y    

def test_truediv():
    x = Variable([1, 2, 3])
    y = Variable([3, 2, 1])
    z = Variable(4)
    a = x + y
    b = a / z
    assert close(b.val, np.matrix([[1], [1], [1]]))
    assert close(b.grad[x], np.eye(3) * 0.25)
    assert close(b.grad[y], np.eye(3) * 0.25)
    assert close(b.grad[z], -np.matrix([[0.25], [0.25], [0.25]]))
    c = b / 0.25
    assert close(c.val, np.matrix([[4], [4], [4]]))
    assert close(c.grad[x], np.eye(3))
    assert close(c.grad[y], np.eye(3))
    assert close(c.grad[z], -np.matrix([[1], [1], [1]]))
    d = 4 / z
    assert close(d.val, 1)
    assert close(d.grad[z], -0.25)

def test_truediv_exception():
    x = Variable([1, 2, 3])
    y = Variable([3, 2, 1])
    z = Variable(0)
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
        b = a / np.matrix([1, 1])
    with pytest.raises(ValueError):
        b = 4.0 / x
    with pytest.raises(TypeError):
        b = a / "g"
    with pytest.raises(TypeError):
        a = True / y

def test_pow():
    x = Variable([1, 2, 3])
    y = Variable([3, 2, 1])
    z = Variable(2)
    a = x + y
    b = z ** 2
    assert close(b.val, 4)
    assert close(b.grad[z], 4)
    b = a ** 2
    assert close(b.val, np.matrix([[16], [16], [16]]))
    assert close(b.grad[x], np.eye(3) * 8)
    assert close(b.grad[y], np.eye(3) * 8)

def test_pow_exception():
    x = Variable([1, 2, 3])
    y = Variable([3, 2, 1])
    z = Variable(2)
    a = x + y
    with pytest.raises(TypeError):
        b = a ** "g"

def test_rpow():
    z = Variable(2)
    b = 2 ** z
    assert close(b.val, 4)
    assert close(b.grad[z], np.log(2) * 4)

def test_rpow_exception():
    x = Variable([1, 2, 3])
    y = Variable([3, 2, 1])
    z = Variable(2)
    a = x + y
    with pytest.raises(TypeError):
        b = "g" ** y
    with pytest.raises(ValueError):
        b = 2 ** a

def test_neg():
    x = Variable([1, 2, 3])
    y = Variable([3, 2, 1])
    z = Variable(4)
    a = x + y
    b = a / z
    c = -b
    assert close(c.val, -np.matrix([[1], [1], [1]]))
    assert close(c.grad[x], -np.eye(3) * 0.25)
    assert close(c.grad[y], -np.eye(3) * 0.25)
    assert close(c.grad[z], np.matrix([[0.25], [0.25], [0.25]]))

test_create_variable()
test_add()
test_add_exception()
test_sub()
test_sub_exception()
test_mul()
test_mul_exception()
test_truediv()
test_truediv_exception()
test_pow()
test_pow_exception()
test_rpow()
test_rpow_exception()
test_neg()
