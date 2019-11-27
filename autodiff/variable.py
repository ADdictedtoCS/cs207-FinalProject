import numpy as np
from autodiff.utils import get_right_shape
#from variable import Variable

class Variable:
    """ 
    #TODO: Write some documentation about that.
    Variable class carry the information flow within the computational graph. 
    Data attribute, Gradient attribute.
    Data represents the evaluation point and gradient is the gradient held 
    by the variable. By default = 1. 
    """
    def __init__(self, val, grad=None): 
        self.val = get_right_shape(val)
        #grad = np.ones((len(self.val), ))
        #We now assume that grad is a n-dimensional element, where n=len(val)
        if grad is None:
           self.grad = np.eye(self.val.shape[0])
        else:
            #self.grad = get_right_shape(grad) 
            self.grad = grad
        print('Initializing the gradient with {}'.format(self.grad))
    
    def __repr__(self):
        return "Value: {}\nGradient: {}".format(self.val, self.grad)
    
    def __add__(self, other):
        if isinstance(other, Variable):
            out_val = self.val + other.val
            out_grad = self.grad + other.grad
            return Variable(val=out_val, grad=out_grad)
        else:
            new_val = get_right_shape(other)
            out_val = self.val + new_val
            out_grad = self.grad
            return Variable(val=out_val, grad=out_grad)

    def __mul__(self, other):
        #Multi-dim: should be np.dot
        #male_sure_shape(self,other)
        if isinstance(other, Variable):
            out_val = self.val * other.val
            out_grad = self.grad * other.val + self.val * other.grad
            return Variable(val=out_val, grad=out_grad)
        else:
            new_val = get_right_shape(other)
            out_val = self.val * new_val
            out_grad = self.grad * new_val
            return Variable(val=out_val, grad=out_grad)
    
    def __radd__(self, other):
        new_val = get_right_shape(other)
        out_val = self.val + new_val
        out_grad = self.grad
        return Variable(val=out_val, grad=out_grad)
    
    def __rmul__(self, other):
        new_val = get_right_shape(other)
        out_val = self.val * new_val
        out_grad = self.grad * new_val 
        return Variable(val=out_val, grad=out_grad)

    def __sub__(self, other):
        if isinstance(other, Variable):
            out_val = self.val - other.val
            out_grad = self.grad - other.grad
            return Variable(val=out_val, grad=out_grad)
        else:
            new_val = get_right_shape(other)
            out_val = self.val - get_right_shape(other)
            out_grad = self.grad
            return Variable(val=out_val, grad=out_grad)

    def __truediv__(self, other):
        #Multi-dim: should be np.dot
        #male_sure_shape(self,other)
        #TODO-1: Make sure the other element is non-zero, Write utils.
        #TODO-2: Extension to vector/multi-dim
        if isinstance(other, Variable):
            if abs(other.val) < 1e-4:
                raise ValueError("Divided by 0!") 
            out_val = self.val / other.val
            out_grad = (self.grad * other.val - self.val * other.grad) / (other.val ** 2)
            return Variable(val=out_val, grad=out_grad)
        else: 
            new_val = get_right_shape(other)
            if abs(new_val) < 1e-4:
                raise ValueError("Divided by 0!")
            out_val = self.val / new_val
            out_grad = self.grad / new_val
            return Variable(val=out_val, grad=out_grad)

    def __rsub__(self, other):
        out_val = get_right_shape(other) - self.val
        out_grad = -self.grad
        return Variable(val=out_val, grad=out_grad)
 
    def __rtruediv__(self, other):
        new_val = get_right_shape(other)
        if abs(self.val) < 1e-4:
            raise ValueError("Divided by 0!")
        out_val = new_val / self.val
        out_grad = -new_val * self.grad / (self.val ** 2)
        return Variable(val=out_val, grad=out_grad)

    def __pow__(self, other):
        new_val = get_right_shape(other)
        if self.val <= 0:
            raise ValueError("Power base cannot be smaller than 0!")
        out_val = self.val ** new_val
        out_grad = new_val * (self.val ** (new_val - 1)) * self.grad
        return Variable(val=out_val, grad=out_grad)

    def __rpow__(self, other):
        new_val = get_right_shape(other)
        # Change later for vector variables
        if new_val <= 0:
            raise ValueError("Power base cannot be smaller than 0!")
        out_val = new_val ** self.val
        out_grad = np.log(new_val) * (new_val ** self.val) * self.grad
        return Variable(val=out_val, grad=out_grad) 
    
    def __neg__(self):
        out_val = -self.val
        out_grad = -self.grad
        return Variable(val=out_val, grad=out_grad)
    
#def dot_(e,X):
    #Assumes e is a (N,1) vector


if __name__ == "__main__":
    import autodiff.function as F
    X = [1,2,3,4]
    e1 = np.array([[0,1,0,0], 
    [1,0,0,0]]).T #4,1
    X = Variable(X)
    print(X.grad.shape, X.val.shape)

    print(X)
    dot = F.Dot(e1)
    y = dot(X)

    
    
    
    
    
    
    u = np.dot(e1.T, X)
    print(u, type(u), len(u))
    print(type(u[0]), type(u[0][0]), u[0][0].val)
    #print(X)
    #Y = F.exp(X)
    #print(Y)
    """FIrst step- We assume that the output is one-d
    # Change the initialization of the gradient.
    #np.dot(st, variable actually returns 
    """
    
