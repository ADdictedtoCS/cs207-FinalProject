import numpy as np
from variable import Variable

"""
- Function object, when called on a variable, returns a new variable with the transformed 
value and gradient.

- Base class that implements the chain rule.

- multiplication, addition, power are implemented in the class Variable. 
It can stay that way or we can consider subclasses such as add(Function), mul(Function)

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
        
    def get_val(self, x):
        return np.exp(x)
    
    def get_grad(self, x):
        return np.exp(x)
    
    #def __repr__(self):
    ##    #Use in base class instead
     #   return 'autodiff.exp'

class Sinus(Function):
    def __init__(self):
        super(Sinus, self).__init__()

    def get_val(self, x):
        return np.sin(x)

    def get_grad(self, x):
        return np.cos(x)

    def __repr__(self):
        #Use in base class instead
        return 'autodiff.sin'

class slice(Function)


def my_func(x):
    y = exp(x)
    #z  = sin(y)
    print('y', y)
    #print('z', z)
    return y


exp = Exponent()
sin = Sinus()


if __name__ == '__main__':
    x = Variable(5)
    #####
    y = exp(x)
    print(y)
    print("HERE BRO", type(y))
    print(type(exp), type(Exponent))

    print(type(Exponent()))
    print(type())
    #y = fn(x)
    print(fn.get_val(x.val))
    print(fn.get_grad(x.val))
    print(np.dot(fn.get_grad(x.val), x.grad))
    #print(y.val, y.grad)
    try: 
        fn(x)
        print(fn(x))
    except Exception as e:
        print(e)
    try:
        y = fn(x)
        new_fn = sin()
        z = new_fn(y)
        print(y)
        print(z)
    except Exception as e:
        print(e)
    
    y=exp(x)

    print("Y", y)
    z = sin(y)
    print("Z", z)


    print('FINAL')
    x = Variable(1)
    type(x)
    z = my_func(x)
    print("final z", z)
    #z.val


    
    
