import pytest
import numpy as np
import autodiff.utils as utils

def test_no_nan_inf_value():
    # ==========================================
    # Test for an input value that is inf or nan
    # ============================================
    with pytest.raises(AssertionError):
        #NaN
        utils._no_nan_inf(np.nan)
        utils._no_nan_inf(float('nan'))
        #Inf
        utils._no_nan_inf(np.inf)
        utils._no_nan_inf(float('inf'))
        # Minus Inf
        utils._no_nan_inf(float('-inf'))

def test_no_zero_value():
    # ==========================================
    # Test for an input value that is inf or nan
    # ============================================
    with pytest.raises(AssertionError):
        utils._no_zero(np.array([1,1,0]))

def test_reshape_float_types():
    # ==========================================
    # Test for an incorrect 'scalar' input
    # ============================================
    with pytest.raises(TypeError):
        #string
        utils.reshape_float('hello')
        #list
        utils.reshape_float([1])

def test_reshape_float_value():
    # ============================================
    # Test to assert that the output is of the type array, with the right np.float64 type
    # ============================================
    #Scalar float
    x = 8.0
    out_x = utils.reshape_float(x)
    assert type(out_x) == np.ndarray and out_x.dtype == np.float64
    #Int float
    x = int(8)
    out_x = utils.reshape_float(x)
    assert type(out_x) == np.ndarray and out_x.dtype == np.float64

def test_reshape_array_dimensions():
    # ============================================
    # Test whether we can reshape a mispecified array into the desired dimension
    # ============================================
    with pytest.raises(TypeError):
        #2x2 matrix
        utils.get_right_shape(np.array([[3, 5], [7, 9]]))
        #array with non np.float64
        x = (1, [44, 'Hey'])
        utils.get_right_shape(np.array(x))
        x = (1, [44, 32.])
        utils.get_right_shape(np.array(x))

#def test_reshape_array_types():
    # ============================================
    # Test whether we can reshape a mispecified array into the desired dimension/type
    # ============================================
#    with pytest.raises(TypeError):
#        x = (10, [145, 26])
        #array with non np.float64
        #x = [1, [44, 'Hey']]
        #utils.get_right_shape(np.array(x))
        #x = [1, [44, 32.]]
        #utils.get_right_shape(np.array(x))
#        utils.get_right_shape(x)

def test_get_right_shape_types():
    # ====================================
    # Test the last utils handling list, tuple, array
    # =====================================
    with pytest.raises(TypeError):
        #Bool
        utils.get_right_shape(True)
        utils.get_right_shape(False)
        # Dict
        utils.get_righ_shape({'x':3})

def test_get_right_shape_result():
    # ====================================
    # Test the last utils handling list, tuple, array
    # =====================================
    correct_x = np.array([33., 2.], dtype=np.float64)
    #List
    assert (utils.get_right_shape([33, 2]) == correct_x).all()
    #Tuple
    assert (utils.get_right_shape((33, 2)) == correct_x).all()
    #List of list mispecified
    assert (utils.get_right_shape([[33, 2]]) == correct_x).all()
    #1-d matrix
    assert (utils.get_right_shape(np.array([33, 2])) == correct_x).all()
    #Vector
    assert (utils.get_right_shape(np.array([[33, 2]])) == correct_x).all()

####test_reshape_array_types()

#test_reshape_array_dimensions()

#test_reshape_float_value()
#test_reshape_float_types()

#test_no_zero_value()
#test_no_nan_inf_value()

#test_get_right_shape_types()
#test_get_right_shape_result()

