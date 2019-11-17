import numpy as np

#TODO-Implement the get_right shape for a list

def get_right_shape(x):
    """
    Function to make sure the input is of the correct
    type for the definition of a variable.
    If np.ndarray is two-dimensional, make it one dimensional.
    """
    #TODO-Handle complex and other weird types
    numeric_type = isinstance(x, (int, float)) and not isinstance(x, bool)
    if numeric_type:
        return reshape_float(x)
    elif isinstance(x, np.ndarray):
        return reshape_array(x)
    elif isinstance(x, list):
        return reshape_array(np.array(x))
    elif isinstance(x, tuple):
        return reshape_array(np.array(x))
    else:
        message = "Input is type {} and should be float, np.ndarray, list or tuple".format(type(x))
        raise TypeError(message)
        
def reshape_array(x):
    """Assume x is a np.ndarray"""
    try:
        out_x = x.reshape(-1,)
    except Exception as e: 
        message = "Evaluation point needs to be one-dimensional \
            found the following problem: {}".format(e)
        raise TypeError(message)
    #We can also loop through an input and make sure that it's made of numeric types.
    try:
        out_x.dtype = np.float64
    except Exception as e:
        raise TypeError("The input contained some values that we could not convert to floating point numbers")
    
    if len(out_x) > max(x.shape): #We have a reshape a matrix into an array. Not wanted.
        raise TypeError("Input can not be  matrix. Input's original shape: {}".format(x.shape))
    #Final assert
    assert (isinstance(out_x, np.ndarray)) and (len(out_x.shape)==1)
    return out_x

def reshape_float(x):
    """Assume x is a float.
    Should work with an int.
    Reshape into a 1-dim np.ndarray"""
    try:
        out_x = np.array([float(x)])
    except Exception as e:
        message = "Evaluation point needs to be one-dimensional \
            found the following problem: {}".format(e)
        raise TypeError(message)
    assert (isinstance(out_x, np.ndarray)) and (len(out_x.shape) == 1)
    return out_x

#TODO-THEO. or not needed with the array.
#Actually handled with the array. 
def reshape_list(x):
    return None

def _no_nan_inf(x):
    assert not np.isnan(x), "Found a nan element"
    assert not np.isinf(x), "Found an inf element"

def _no_zero(x):
    assert not np.any(x), "Found a zero element"
    
