import numpy as np
from utils import get_right_shape

class Variable:
    """ 
    #TODO: Write some documentation about that.
    Variable class carry the information flow within the computaitonal graph. 
    Data attribute, Gradient attribute.
    Data represents the evaluation point and gradient is the gradient held 
    by the variable. By default = 1. 
    """
    def __init__(self, val, grad=1.): 
        self.val = get_right_shape(val)
        #grad = np.ones((len(self.val), ))
        self.grad = get_right_shape(grad)
    
    def __repr__(self):
        return "Value: {}\nGradient: {}".format(self.val, self.grad)
    
    def __add__(self, other):
        out_val = self.val + other.val
        out_grad = self.grad + other.grad
        return Variable(val=out_val,
        grad=out_grad)

    def __mul__(self, other):
        #Multi-dim: should be np.dot
        #male_sure_shape(self,other)
        out_val = self.val * other.val
        out_grad = self.grad * other.val + self.val * other.grad
        return Variable(val=out_val, grad=out_grad)
    
    #TODO-T,J
    def __radd__():
        return None
    
    #TODO-T,J
    def __rmul__():
        return None
    
    #TODO-T,J
    def __pow__():
        return None
    
    #TODO-T,J
    def __neg__():
        return None
    
    #def __getitem__(self):
        #TODO
        return None
    #@classmethod
    #def _transform(cls, func):
    #    #TODO: assert type(func) == autodiff.Function:
    #    return cls()

if __name__ == "__main__":
    x = Variable(int(5))
    y = Variable(8)
    print(x)
    print(type(x.val), type(x.grad))
    z = x*y
    print(z)
    z = z*x
    print(z)
