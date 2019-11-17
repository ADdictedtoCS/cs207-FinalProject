#import numpy as np
import sys  # TODO-External dependecy. How to do it otherwise??
import pytest
sys.path.append("..")
#from .. import autodiff
from autodiff.utils import get_right_shape, reshape_float, reshape_array, reshape_array, _no_nan_inf, _no_zero

def first_tests():
    x = np.array([[12, 14]])
    print(type(x))
    try:
        isinstance(x, list)
    except Exception as e:
        print(e)

    #x = x.reshape(1)
    print(x.shape)
    print(x[0])
    X = np.array(x)
    print(x.shape)
    print(type(x))
    ####
    print('Start')
    x = np.array([[12]])
    out_x = reshape_array(x)
    print(x.shape)
    print(out_x.shape)
    ####
    x = np.array([[12]])
    out_x = reshape_array(x)
    print(x.shape)
    print(out_x.shape)
    ###
    ###
    try:
        x = np.array([[12, 14]])
        out_x = reshape_array(x)
    except Exception as e:
        print(e)
    print('DOne')
    x = int(12)
    XX = np.array(8)
    print("TYPE XX", type(XX))
    XX = XX.reshape(1,)
    print("TYPE XX", type(XX))
    out_x = reshape_float(x)
    print(out_x.shape)
    print(out_x)
    print(type(out_x))
    x = float(14)
    out_x = reshape_float(x)
    print(out_x.shape)
    print(out_x)

def test_function(fn, x):
    try:
        fn(x)
        print('Passed test')
    except Exception as e:
        print("Raised the following exception", e)
    return None



x = np.array([[12, 14]])
print(x.shape, len(x))
print(x)
out_x = get_right_shape(x)
print(type(out_x), out_x.shape)
#x = np.array([[12]])

#1-List of list
print("List of list test")
x = [[123, 34], [34, 26, 38]]
test_function(get_right_shape, x)

#2-List of list where the second dimension is null
print("Handle the user mispecified list")
x = [[123, 34]]
test_function(get_right_shape, x)

#3-Correct list
x = [1000]
print('Simple one-dimensional list')
test_function(get_right_shape, x)

#4-Matrix-form Array
x = np.array([[12, 14], [18, 36]])
print(x, x.dtype)
new_x = x
new_x.dtype = np.float64
print(new_x, new_x.dtype)
print("Matrix input")
test_function(get_right_shape, x)

#5-Mispecified array
x = np.array([[12]])
print("BBB", x.dtype)
print("Handle the user mispecified array")
test_function(get_right_shape, x)

#6-Mispecified array
x = np.array([1000])
print("AAA", x.dtype)
#TODO-Convert the array into "DTYPE-float"
print('Simple one-dimensional array')
test_function(get_right_shape, x)

#7-Float
x = 1.
test_function(get_right_shape, x)

#8-Int
x = int(8)
test_function(get_right_shape, x)
out_x = get_right_shape(x)
try:
    out_x.dtype = np.float64
    print("OK for changing the type")
    print(out_x.dtype)
except Exception as e:
    print(e)
#9-Bool
x = True
test_function(get_right_shape, x)
x = False
test_function(get_right_shape, x)

#10-Tuple
x = (10, 14)
test_function(get_right_shape, x)

#11-Tuple misp.
print("TEST with weird array")
x = (10, [145, 26])
test_function(get_right_shape, x)
#out_x = get_right_shape(x)
#print(out_x, type(out_x))
#try:
#    out_x.dtype = np.float64
#except Exception as e:
#    print(e)

######TODO-Does not work on everything, examples, this one.
#####Do we need to handle all of those cases ?
###Check the types of the array ?
###Future direction-create our own autodiff.array()?!
### Future-Implement some functions in Cython ?

x = np.nan
test_function(_no_nan_inf, x)

print("NAN Test")
_no_nan_inf(x)
