import numpy as np

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

def get_right_shape(x):
    """
    Transform an input into the array of desired shape ( i.e (N,) ) and type (np.float64).
    Handles list, tuple, array, int and float by calling reshape_array or reshape_float. 
    
    INPUT
    =======
    x: object
    
    OUTPUT
    =======
    - Output of reshape array or reshape float
    or
    - Type error

    """
    #TODO-Handle complex and other weird types
    numeric_type = isinstance(x, (int, float)) and not isinstance(x, bool)
    if numeric_type:
        # return reshape_float(x)
        return float(x)
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
    """
    Assume x is an arbitrary np.ndarray. Reshape it into a (N,) array with np.float64 types.
        Raise Error for matrix and other unconventinoal types. 
    
    INPUTS 
    =====
    x : np.ndarray
        The array to be reshaped
    
    OUTPUTS
    ======
    out_x: np.ndarray
        Reshaped array with null second dimension and composed of np.float64.
    
    EXAMPLE
    =======
    x = np.array([[2,3]])
    out_x = reshape_array(x)
    print(out_x.shape, out_x.dtype)
    (2,) float64
    """

    try:
        # out_x = x.reshape(-1,)
        # out_x = np.reshape(x, (x.shape[0], 1))
        if len(x.shape) == 1:
            out_x = np.reshape(x, (x.shape[0], 1))
        else:
            out_x = x
    except Exception as e: 
        message = "Evaluation point needs to be one-dimensional \
            found the following problem: {}".format(e)
        raise TypeError(message)
    #We can also loop through an input and make sure that it's made of numeric types.
    try:
        out_x = out_x.astype(np.float64)
    except Exception as e:
        raise TypeError("The input contained some values that we could not convert to floating point numbers")
    
    # if len(x.shape) == 0: #Just a value
    #     assert (isinstance(out_x, np.ndarray)) and (len(out_x.shape) == 1)
    #     return out_x
    # if len(out_x) > max(x.shape): #We have a reshape a matrix into an array. Not wanted.
    #     raise TypeError("Input can not be  matrix. Input's original shape: {}".format(x.shape))
    # #Final assert
    # assert (isinstance(out_x, np.ndarray)) and (len(out_x.shape)==1)
    # return np.matrix(out_x)
    return np.ndarray(out_x.shape, dtype=float, buffer=out_x)

# def reshape_float(x):
#     """Assume x is a float. Create a np.array with a np.float64 value.
#         Handles type int.
#     Out: 1-dim np.ndarray
#     """
#     try:
#         # out_x = np.array([float(x)])
#         out_x = x
#     except Exception as e:
#         message = "Evaluation point needs to be one-dimensional \
#             found the following problem: {}".format(e)
#         raise TypeError(message)
#     try:
#         out_x = out_x.astype(np.float64)
#     except Exception as e:
#         raise TypeError(
#             "The input contained some values that we could not convert to floating point numbers")
#     # assert (isinstance(out_x, np.ndarray)) and (len(out_x.shape) == 1)
#     return out_x

def _no_nan_inf(x):
    """
    Raise AssertionError if if an element is nan/inf.
    """
    assert not np.isnan(x), "Found a nan element"
    assert not np.isinf(x), "Found an inf element"

def _no_zero(x):
    """
    Raise AssertionError if if an element is zero.
    """
    assert not np.any(x), "Found a zero element"
    
