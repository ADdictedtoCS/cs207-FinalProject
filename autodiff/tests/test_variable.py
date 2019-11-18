#Sequence of tests for the variables operations.
#TODO-Junzhi

import pytest
import numpy as np
from autodiff.variable import Variable

def test_create_variable():
    x = Variable(0)
    assert x.val == 0 and x.grad == 1
    x = Variable([1], [2])
    assert x.val == 1 and x.grad == 2
    x = Variable(2.2, 3.3)
    assert x.val == 2.2 and x.grad == 3.3
    x = Variable(np.ndarray((1), dtype=float, buffer=np.array([3])), np.ndarray((1), dtype=float, buffer=np.array([2])))
    assert x.val == 3 and x.grad == 2
    x = Variable((4), (5))
    assert x.val == 4 and x.grad == 5

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
    x = Variable(1)
    y = x + x
    assert y.val == 2 and y.grad == 2
    z = y + x
    assert z.val == 3 and z.grad == 3
    a = z + 4.4
    assert a.val == 7.4 and a.grad == 3
    b = 3.2 + z
    assert b.val == 6.2 and b.grad == 3
    c = b + a
    assert c.val == 13.6 and c.grad == 6

def test_add_exception():
    x = Variable(2)
    with pytest.raises(TypeError):
        y = x + "g"
    with pytest.raises(TypeError):
        y = True + x

def test_sub():
    x = Variable(2)
    y = x + x
    z = y + x
    a = z + 4.4
    b = a - x
    assert b.val == 8.4 and b.grad == 2
    c = a - 2.1
    assert c.val == 8.3 and c.grad == 3
    d = 3 - a
    assert d.val == -7.4 and d.grad == -3

def test_sub_exception():
    x = Variable(2)
    y = x + x
    z = y + x
    a = z + 4.4
    with pytest.raises(TypeError):
        b = x - "g"
    with pytest.raises(TypeError):
        b = True - x

def test_mul():
    x = Variable(3)
    y = x + x + 4
    z = 7 - x
    a = y * z
    assert a.val == 40 and a.grad == -2
    b = a * 4
    assert b.val == 160 and b.grad == -8
    c = 3 * a
    assert c.val == 120 and c.grad == -6

def test_mul_exception():
    x = Variable(3)
    y = x + x + 4
    with pytest.raises(TypeError):
        z = y * "g"
    with pytest.raises(TypeError):
        z = False * y    

def test_truediv():
    x = Variable(3)
    y = x + x + 4
    z = 7 - x
    a = y / z
    assert a.val == 2.5 and a.grad == 18.0/16.0
    b = y / 5
    assert b.val == 2 and b.grad == 0.4
    c = 4 / y
    assert c.val == 0.4 and c.grad == 2 

def test_truediv_exception():
    x = Variable(3)
    y = x + x + 4
    z = 3 - x
    with pytest.raises(ValueError):
        a = y / z
    with pytest.raises(ValueError):
        a = y / 0.0
    with pytest.raises(ValueError):
        a = 4.0 / z
    with pytest.raises(TypeError):
        a = y / "g"
    with pytest.raises(TypeError):
        a = True / y

def test_pow():
    x = Variable(3)
    y = x + x + 4
    z = y ** 3
    assert z.val == 1000 and z.grad == 600

def test_pow_exception():
    x = Variable(4)
    y = x + x + 4
    with pytest.raises(TypeError):
        z = y ** "g"

def test_rpow():
    x = Variable(3)
    y = x + x + 4
    z = 2 ** y
    assert z.val == 1024 and np.abs(z.grad - np.log(2) * 2048) < 1e-4

def test_rpow_exception():
    x = Variable(4)
    y = x + x + 4
    with pytest.raises(TypeError):
        z = "g" ** y
    with pytest.raises(ValueError):
        z = 0 ** y
    with pytest.raises(ValueError):
        z = -2 ** y

def test_neg():
    x = Variable(4)
    y = x + x + 4
    z = -y
    assert z.val == -12 and z.grad == -2

test_create_variable()
test_create_variable_exception()
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
