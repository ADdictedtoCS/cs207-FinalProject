import numpy as np
from autodiff.variable import Variable
from autodiff.utils import get_right_shape

"""
Function class is the base class that implements the chain rule and other basic methods.
A Function object takes a Variable object, and returns a new Variable with the transformed value and gradient
The elementary functions that are currently implemented are exp, sin, cos, and tan.

Multiplication, addition, power are implemented in the class Variable. 
It can stay that way or we can consider subclasses such as add(Function), mul(Function)

The Different classes are instantiated so that we can easily import.
Example: import function as F
x=Variable, y = F.exp(x)
"""

class Function:
    """
   The get_grad and get_val methods are not implemented for this base class 
   but get_grad implements calculation of derivative, and get_val implements calculation of value
   for the elementary functions which are subclasses of function   
    """

    def get_grad(self, x):
        raise NotImplementedError

    def get_val(self, x):
        raise NotImplementedError

    def __repr__(self):
        """    
        RETURNS
        ========
        string: contains a printable representation of an Function object
    
        """
        return '{}'.format(type(self))

    def __call__(self, x):
        """Implements the chain rule.
        INPUTS
        =======
        x: autodiff.Variable holding a val and grad
    
        RETURNS
        ========
        autodiff.Variable: updated Variable after chain rule was applied 
        """
        out_val = self.get_val(x.val)
        out_grad = np.dot(self.get_grad(x.val), x.grad)
        return Variable(val=out_val, grad=out_grad)
    
class Exponent(Function):
    """Implements calculation of value and derivative of Exponential function
        Overloads get_val and get_grad from the Function class
        
    """   
    def get_val(self, x):        
        return np.exp(x)
    
    def get_grad(self, x):
        return np.exp(x)
    
class Sinus(Function):
    """Implements calculation of value and derivative of Sine function
        Overloads get_val and get_grad from the Function class
        
    """   
    def get_val(self, x):
        return np.sin(x)

    def get_grad(self, x):
        return np.cos(x)

class Cosinus(Function):
    """Implements calculation of value and derivative of Cosine function
        Overloads get_val and get_grad from the Function class
    """   
    def get_val(self, x):
        return np.cos(x)

    def get_grad(self, x):
        return - np.sin(x)

class Tangent(Function):
    """Implements calculation of value and derivative of Tangent function
        Overloads get_val and get_grad from the Function class
        
    """   
    def get_val(self, x):
        tmp = (x - np.pi / 2) / np.pi
        if abs(tmp - tmp.round()) < 1e-4:
            raise ValueError("Value not in the domain!")
        return np.tan(x)

    def get_grad(self, x):
        tmp = (x - np.pi / 2) / np.pi
        return 1./np.cos(x)**2

class Dot(Function):
    """
    assumes e is a (N,p) array
    """
    def __init__(self, e):
        self.e = e

    def get_val(self, x):
        return np.dot((self.e).T, x) #p,N x N, = p,

    def get_grad(self, x):
        return (self.e).T #p,N
    
#class Dot_:
#See if we want to include that
#    def __call__(self, e):
#        return Dot(e)(x) 

def generate_base(N):
    """Function to generate the canonical basis of R^{N}
    """
    Id = np.eye(N)
    basis = {'e{}'.format(i): Id[i].reshape(-1,1) for i in range(N)}
    return basis

def unroll(X):
    """
    Assumes X is a autodiff.variable
    X.val is (N,) array
    """
    output = []
    N = X.val.shape[0]
    base = generate_base(N)
    for e in base.values():
        output.append(Dot(e)(X))
    return output
    
exp = Exponent()
sin = Sinus()
cos = Cosinus()
tan = Tangent()


if __name__ == "__main__":
    #=====================
    #DEMO
    #===================
    from autodiff.variable import Variable 
    X = Variable(np.array([1,5,10]))
    x,y,z = unroll(X)
    print(x,y,z)
    out = exp(x) + cos(y)
    out += x
    #=============
    #Check whether it matches what we hoped ?
    #===========
    print('Expected value', np.exp(X.val[0])+np.sin(X.val[1]))
    print('Expected gradients', np.exp(X.val[0])+1, -np.sin(X.val[1]), 0)