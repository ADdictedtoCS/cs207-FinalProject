import numpy as np
from autodiff.variable import Variable
from autodiff.utils import get_right_shape
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
        Output:  autodiff.Variable type holding the val, grad of the transformed variable
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
#dot_ = Dot()
   
        

if __name__ == "__main__":
    from autodiff.variable import Variable
    