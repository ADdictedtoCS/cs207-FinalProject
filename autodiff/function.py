import numpy as np
from variable import Variable
from utils import get_right_shape
"""
- Function object, when called on a variable, returns a new variable with the transformed 
value and gradient.

- Base class that implements the chain rule and other basic methods.
The subclasses overload get_val and get_grad only

- multiplication, addition, power are implemented in the class Variable. 
It can stay that way or we can consider subclasses such as add(Function), mul(Function)

-The Different classes are instantiated so that we can easily import.
Example: import function as F
x=Variable, y = F.exp(x)
"""

class Function:
    def __init__(self):
        return None

    def get_grad(self, x):
        #Works on array
        raise NotImplementedError

    def get_val(self, x):
        #Works on array
        raise NotImplementedError

    def __repr__(self):
        #TODO
        return '{}'.format(type(self))

    #Works on AD.Variable
    def __call__(self, x):
        """
        Implements the chain rule.
        Input: autodiff.Variable type holding a val and grad
        Output:  autodiff.Variable type holding the val, grad of the transormed variable
        """
        out_val = self.get_val(x.val)
        out_grad = np.dot(self.get_grad(x.val), x.grad)
        return Variable(val=out_val, grad=out_grad)
    
class Exponent(Function):
    """Exponential"""    
    def get_val(self, x):
        return np.exp(x)
    
    def get_grad(self, x):
        return np.exp(x)
    
class Sinus(Function):
    """Sine"""
    def get_val(self, x):
        return np.sin(x)

    def get_grad(self, x):
        return np.cos(x)

class Cosinus(Function):
    """ Cosine"""
    def get_val(self, x):
        return np.cos(x)

    def get_grad(self, x):
        return - np.sin(x)

class Tangent(Function):
    """ Tangent"""
    def get_val(self, x):
        return np.tan(x)

    def get_grad(self, x):
        return 1./np.cos(x)**2


def my_func(x):
    x = exp(x)
    y = cos(x)
    z = sin(x)
    return y + z

exp = Exponent()
sin = Sinus()
cos = Cosinus()
tan = Tangent()






    


    
    
