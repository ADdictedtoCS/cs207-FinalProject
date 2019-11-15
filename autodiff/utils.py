import numpy as np


#TODO-Implement the get_right shape for a list

def get_right_shape(x):
    """
    Function to make sure the input is of the correct
    type for the definition of a variable.
    If np.ndarray is two-dimensional, make it one dimensional.
    """
    #TODO=Handle complex and other weird types
    numeric_type = isinstance(x, (int, float, complex)) and not isinstance(x, bool)
    if numeric_type:
        return reshape_float(x)
    elif isinstance(x, np.ndarray):
        return reshape_array(x)
    else:
        message = "Input is type {} and should be float or np.ndarray"
        raise TypeError(message)
        
def reshape_array(x):
    """Assume x is a np.ndarray"""
    try:
        out_x = x.reshape(-1,)
    except Exception as e: 
        message = "Evaluation point needs to be one-dimensional \
            found the following problem: {}".format(e)
        raise TypeError(message)
    
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

#TODO-
def reshape_list(x):
    return None

def first_tests():
    x = np.array([[12,14]])
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
    x=int(12)
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


if __name__ == "__main__":
    x = np.array([[12,14]])
    print(x.shape, len(x))
    print(x)
    out_x = get_right_shape(x)
    print(type(out_x), out_x.shape)
    #x = np.array([[12]])
    ##out_x = get_right_shape(x)
    #print(type(out_x), out_x.shape)
    x = np.array([[12, 14], [18, 36]])
    try:
        out_x = get_right_shape(x)
        print(type(out_x), out_x.shape)
    except Exception as e:
        print('CAUGHT', e)
    
