#Sequence of tests for the variables operations.

import pytest
import numpy as np
from autodiff.variable import Variable
from autodiff.utils import *

def test_create_variable():
    X = Variable([1, 2, 3])
    assert close(X.val, np.asarray([[1], [2], [3]]))
    assert close(X.grad, np.eye(3))
    var_list = X.unroll()
    x = var_list[0]
    assert close(x.val, 1)
    assert close(x.grad, np.asarray([[1, 0, 0]]))
    var_list = X.unroll([2, 1])
    x = var_list[0]
    assert close(x.val, np.asarray([[1], [2]]))
    assert close(x.grad, np.asarray([[1, 0, 0], [0, 1, 0]]))
    y = Variable(1)
    var_list = y.unroll()
    x = var_list[0]
    assert close(x.val, 1)
    assert close(x.grad, 1)
    var_list = y.unroll([1])
    x = var_list[0]
    assert close(x.val, 1)
    assert close(x.grad, 1)

def test_unroll_exception():
    X = Variable([1, 2, 3])
    with pytest.raises(TypeError):
        var_list = X.unroll(1)
    with pytest.raises(ValueError):
        var_list = X.unroll(['a'])
    with pytest.raises(ValueError):
        var_list = X.unroll([0, 1, 2])
    with pytest.raises(ValueError):
        var_list = X.unroll([2, 1, 1])
    y = Variable(1)
    with pytest.raises(ValueError):
        var_list = y.unroll([1, 1])

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
    X = Variable([1, 2, 2, 1])
    var_list = X.unroll([2, 2])
    x = var_list[0]
    y = var_list[1]
    z = x + y
    assert close(z.val, np.asarray([[3], [3]]))
    assert close(z.grad, np.asarray([[1, 0, 1, 0], [0, 1, 0, 1]]))
    a = z + [4.4, 3.3]
    b = a + z
    assert close(b.val, np.asarray([[10.4], [9.3]]))
    assert close(b.grad, np.asarray([[2, 0, 2, 0], [0, 2, 0, 2]]))
    c = [4.4, 3.3] + b
    assert close(c.val, np.asarray([[14.8], [12.6]]))
    assert close(c.grad, np.asarray([[2, 0, 2, 0], [0, 2, 0, 2]]))
    
def test_add_exception():
    x = Variable([1, 2, 3])
    with pytest.raises(TypeError):
        y = x + "g"
    with pytest.raises(TypeError):
        y = True + x

def test_sub():
    X = Variable([1, 2, 2, 1])
    var_list = X.unroll([2, 2])
    x = var_list[0]
    y = var_list[1]
    z = y + x
    a = z + [4.4, 3.3]
    b = a + z
    c = b - x
    assert close(c.val, np.asarray([[9.4], [7.3]]))
    assert close(c.grad, np.asarray([[1, 0, 2, 0], [0, 1, 0, 2]]))
    d = 3 - c
    assert close(d.val, np.asarray([[-6.4], [-4.3]]))
    assert close(d.grad, np.asarray([[-1, 0, -2, 0], [0, -1, 0, -2]]))
    e = c - 2.1
    assert close(e.val, np.asarray([[7.3], [5.2]]))
    assert close(e.grad, np.asarray([[1, 0, 2, 0], [0, 1, 0, 2]]))

def test_sub_exception():
    X = Variable([1, 2, 2, 1])
    var_list = X.unroll([2, 2])
    x = var_list[0]
    y = var_list[1]
    z = y + x
    a = z + [4.4, 3.3]
    b = a + z
    with pytest.raises(TypeError):
        c = b - "g"
    with pytest.raises(TypeError):
        c = True - b

def test_mul():
    X = Variable([1, 2, 2, 1, 4])
    var_list = X.unroll([2, 2, 1])
    x = var_list[0]
    y = var_list[1]
    z = var_list[2]
    a = x + y
    b = a * z
    assert close(b.val, np.asarray([[12], [12]]))
    assert close(b.grad, np.asarray([[4, 0, 4, 0, 3], [0, 4, 0, 4, 3]]))
    c = b * 4
    assert close(c.val, np.asarray([[48], [48]]))
    assert close(c.grad, np.asarray([[16, 0, 16, 0, 12], [0, 16, 0, 16, 12]]))
    d = 2 * b
    assert close(d.val, np.asarray([[24], [24]]))
    assert close(d.grad, np.asarray([[8, 0, 8, 0, 6], [0, 8, 0, 8, 6]]))
    # M = np.eye(3) * 3
    # e = b.__rmul__(M)
    # assert close(e.val, np.asarray([[48], [48], [48]]))
    # assert close(e.grad[x], np.eye(3) * 12)
    # assert close(e.grad[y], np.eye(3) * 12)
    # assert close(e.grad[z], np.asarray([[12], [12], [12]]))
    
def test_mul_exception():
    X = Variable([1, 2, 2, 1, 4])
    var_list = X.unroll([2, 2, 1])
    x = var_list[0]
    y = var_list[1]
    z = var_list[2]
    a = x + y
    with pytest.raises(TypeError):
        b = a * "g"
    with pytest.raises(TypeError):
        b = False * y    

def test_div():
    X = Variable([1, 3, 3, 1, 4])
    var_list = X.unroll([2, 2, 1])
    x = var_list[0]
    y = var_list[1]
    z = var_list[2]
    a = x + y
    b = a / z
    assert close(b.val, np.asarray([[1], [1]]))
    assert close(b.grad, np.asarray([[0.25, 0, 0.25, 0, -0.25], [0, 0.25, 0, 0.25, -0.25]]))
    c = b / 0.25
    assert close(c.val, np.asarray([[4], [4]]))
    assert close(c.grad, np.asarray([[1, 0, 1, 0, -1], [0, 1, 0, 1, -1]]))
    d = 4 / z
    assert close(d.val, 1)
    assert close(d.grad, np.asarray([[0, 0, 0, 0, -0.25]]))

def test_div_exception():
    X = Variable([1, 3, 3, 1, 0])
    var_list = X.unroll([2, 2, 1])
    x = var_list[0]
    y = var_list[1]
    z = var_list[2]
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
    X = Variable([1, 3, 3, 1, 2])
    var_list = X.unroll([2, 2, 1])
    x = var_list[0]
    y = var_list[1]
    z = var_list[2]
    a = x + y
    b = z ** 2
    assert close(b.val, 4)
    assert close(b.grad, np.asarray([[0, 0, 0, 0, 4]]))
    b = a ** 2
    assert close(b.val, np.asarray([[16], [16]]))
    assert close(b.grad, np.asarray([[8, 0, 8, 0, 0], [0, 8, 0, 8, 0]]))
    c = z
    d = z ** c
    assert close(d.val, 4)
    assert close(d.grad, np.asarray([[0, 0, 0, 0, 4 * (np.log(2) + 1)]]))

def test_pow_exception():
    X = Variable([1, 3, 3, 1, 2])
    var_list = X.unroll([2, 2, 1])
    x = var_list[0]
    y = var_list[1]
    z = var_list[2]
    a = x + y
    with pytest.raises(TypeError):
        b = a ** "g"
    with pytest.raises(ValueError):
        b = a ** [3, 3]
    with pytest.raises(ValueError):
        b = a ** z
    with pytest.raises(ValueError):
        b = z ** a

def test_rpow():
    z = Variable(2)
    b = 2 ** z
    assert close(b.val, 4)
    assert close(b.grad, np.log(2) * 4)

def test_rpow_exception():
    X = Variable([1, 3, 3, 1, 2])
    var_list = X.unroll([2, 2, 1])
    x = var_list[0]
    y = var_list[1]
    z = var_list[2]
    a = x + y
    with pytest.raises(TypeError):
        b = "g" ** y
    with pytest.raises(ValueError):
        b = 2 ** a

def test_neg():
    X = Variable([1, 3, 3, 1, 4])
    var_list = X.unroll([2, 2, 1])
    x = var_list[0]
    y = var_list[1]
    z = var_list[2]
    a = x + y
    b = a / z
    c = -b
    assert close(c.val, -np.asarray([[1], [1]]))
    assert close(c.grad, np.asarray([[-0.25, 0, -0.25, 0, 0.25], [0, -0.25, 0, -0.25, 0.25]]))

def test_eq():
    X = Variable([1, 3, 3, 1, 4])
    var_list = X.unroll([2, 2, 1])
    x = var_list[0]
    y = var_list[1]
    z = var_list[2]
    a = x + y
    b = a + [4.4, 3.3]
    c = a - x
    assert c == y
    assert c != x
    c = b - a
    assert c == [4.4, 3.3]
    assert [4.4, 3.3] == c
    assert [1, 1] != c
    assert c != [1, 1]

test_create_variable()
test_unroll_exception()
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
