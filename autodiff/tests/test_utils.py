import pytest
import numpy as np
import autodiff.utils as utils

def test_get_right_shape_types():
    with pytest.raises(TypeError):
        utils.get_right_shape(True)
        utils.get_right_shape(False)
        utils.get_right_shape([4,5])
        utils.get_right_shape((33, 2))
        utils.get_right_shape(np.array([33, 2]))
        utils.get_right_shape(np.array([4, 5]))
        utils.get_right_shape([[33, 2]])

def test_reshape_array_dimension():
    with pytest.raises(TypeError):
        utils.get_right_shape(np.array([[3, 5], [7, 9]]))

test_get_right_shape_types()
test_reshape_array_dimension()
#print('OK WOKR ')

