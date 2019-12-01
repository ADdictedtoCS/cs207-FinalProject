import numpy as np
from autodiff.variable import Variable
from autodiff.utils import get_right_shape
import warnings
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
    assumes e is a (N,p) array.
    Elements of e should not require gradient computations.
    """
    def __init__(self, e):
        self.e = e

    def get_val(self, x):
        return np.dot((self.e).T, x) #p,N x N, = p,

    def get_grad(self, x):
        return (self.e).T #p,N
    
class Dot_(Function):
    """
    User friendly usage of dot. No Need to instantiate the dot. 
    Works on right and left multiplication by a matrix for instance.
    """
    def __init__(self):
        return None

    def __call__(self, e, x):
        try:    
            return Dot(e)(x) 
        except Exception as exc:
            message = "Need to provide a Variable and right shapesTypes and shapes are: {}, {}".format(type(e), type(x))
            assert isinstance(e, Variable), message
            val = Dot(x.T)(e)
            warnings.warn('Matrix multiplication on the right')
            return val
                    
            #if not isinstance(e)
                #val = Dot(x)(e)

def concat(var_list:list):
    """ 
    If x, y variables, it should let the user define conc_x,y = F.concat([x,y]) which is now a multivariate stuff. 
    Assume we have two variables in R^2 and R^3 respectively.
    There are supposed to have the same input space, for instance X^10 so that the gradietns are 10,2 and 10,3 dimensions.
    var_list has to be a list of var. 
    """
    assert len(var_list)>0, 'Can not concatenate an empty list'
    input_dim = var_list[0].grad.shape[1] #grad shape of the first variable in the list
    concat_val, concat_grad = [], []
    for var in var_list:
        assert (var.grad.shape[1] == input_dim, 
        'trying to concatenate variables from a different input')
        concat_val.append(var.val)
        concat_grad.append(var.grad)
        print(var.grad.shape)
    out_val = np.concatenate(concat_val)
    out_grad = np.concatenate(concat_grad, axis=0)
    return Variable(val=out_val, grad=out_grad)
    
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
    Output  a list of smaller variables.
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
dot_ = Dot_()


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

    #====================
    # DEMO with a Linear matrix mulitplication ? 
    #====================
    #X does not change.
    print('Second part')
    e0, e1, e2 = generate_base(X.val.shape[0]).values()
    print(e1)
    out = dot_(e0,X)
    print('Out', out)
    out_2 = dot_(e1, X)
    print('Out_2', out_2)

    matrix = np.array([[4,6,0], [1,0,2]]).T
    matrix_mul = dot_(matrix, X) #2,3 x #3, -> 2,
    print('matrix_mul', matrix_mul)
    #What if we want to expand to a bigger dimension ? 
    new_mm = dot_(matrix.T, matrix_mul) #3,2->2,
    print('New mm', new_mm)

    #print(type(X), 'class autodiff.variable.Variable')
    #print(getattr(X,'type'))
    print(isinstance(X,Variable))
    
    out_x = dot_(X, matrix.T) #3, x 3,2
    print('out_x', out_x)

    new_X = concat([x,y])
    print(X)
    print(new_X)

    full_X = concat([new_X,z])
    print('full_X', full_X)
    #print('EQ?', full_X == X)
    print((X.grad==full_X.grad).all())







