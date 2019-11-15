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
        message = "Input is type {} and should be float, np.ndarray or list".format(type(x))
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
def reshape_list(x):
    return None

def _no_nan_inf(x):
    assert not np.isnan(x), "Found a nan element"
    assert not np.isinf(x), "Found an inf element"

def _no_zero(x):
    assert not np.any(X), "Found a zero element"
    


#First basics tests
if __name__ == "__main__":
    x = np.array([[12,14]])
    print(x.shape, len(x))
    print(x)
    out_x = get_right_shape(x)
    print(type(out_x), out_x.shape)
    #x = np.array([[12]])
 

    #1-List of list
    print("List of list test")
    x = [[123, 34], [34, 26,38]]
    test_function(get_right_shape,x)

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
    x = (10,14)
    test_function(get_right_shape, x)

    #11-Tuple misp.
    print("TEST with weird array")
    x = (10, [145,26])
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




