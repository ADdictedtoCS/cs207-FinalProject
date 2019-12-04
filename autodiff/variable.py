import numpy as np
from autodiff.utils import get_right_shape
import autodiff

class Variable:
    """ 
    A Variable is an object, which carries the information flow
    within the computational graph.
    """
    def __init__(self, val, grad=None): 
        """
        Variables are initialized with a value and a gradient.

         INPUTS
        =======
        val: float, int, 1-D tuple, or 1-D list, required.
            Is the value of the variable. Currently handles numeric and
            1-D types, but will be extended to take multidimensional input
            in the near future.

        grad: float or int, optional. Default value is 1 (the seed).
            Is the gradient of the variable.

        EXAMPLES
        =========
        >>> x = Variable(2.0)
        >>> x = Variable((2))
        >>> x = Variable(np.array([2]))
        """

        # Assure val and grad are correct shape (in preparation for
        # multivariate implementation)
        self.val = get_right_shape(val)
        #grad = np.ones((len(self.val), ))
        #We now assume that grad is a n-dimensional element, where n=len(val)
        if grad is None:
           self.grad = np.eye(self.val.shape[0])
        else:
            #self.grad = get_right_shape(grad) 
            self.grad = grad
        #print('Initializing the gradient with {}'.format(self.grad))
    
    def __repr__(self):
        """ When variables are printed, gives both val and grad"""
        return "Value: {}\nGradient: {}".format(self.val, self.grad)
    
    def __add__(self, other):
        """Implements addition between Variables and other objects, 
            which are either Variables or numeric values. 
    
        INPUTS
        =======
        other: Variable, float, or int.
            The object with which we are adding our Variable.
    
        RETURNS
        ========
        Variable: A new variable whose val and grad are those resulting
            from the summation of our Variable and other.
    
        EXAMPLES
        =========
        >>> x = Variable(2.0)
        >>> y = Variable(4.0)
        >>> z = x + y
        >>> print(z)
        Value: [6.]
        Gradient: [2.]
        """
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
        """Implements multiplication between Variables and other objects,
            which are either Variables or numeric values.
    
        INPUTS
        =======
        other: Variable, float, or int.
            The object with which we are multiplying our Variable.
    
        RETURNS
        ========
        Variable: A new variable whose val and grad are those resulting
            from the multiplication of our Variable and other.
    
        EXAMPLES
        =========
        >>> x = Variable(2.0)
        >>> y = Variable(4.0)
        >>> z = x * y
        >>> print(z)
        Value: [8.]
        Gradient: [6.]
        """
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
        """Implements addition between other objects and Variables.
            See __add__ for reference.
        """
        new_val = get_right_shape(other)
        out_val = self.val + new_val
        out_grad = self.grad
        return Variable(val=out_val, grad=out_grad)
    
    def __rmul__(self, other):
        """Implements multiplication between other objects and Variables.
            See __mul__ for reference.
        """
        new_val = get_right_shape(other)
        out_val = self.val * new_val
        out_grad = self.grad * new_val 
        return Variable(val=out_val, grad=out_grad)

    def __sub__(self, other):
        """Implements subtraction between Variables and other objects.
            See __add__ for reference.
        """
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
        """Implements division between Variables and other objects.
            See __mul__ for reference.
        """
        
        #Multi-dim: should be np.dot
        #make_sure_shape(self,other)
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
        """Implements subtraction between other objects and Variables.
            See __sub__ for reference.
        """
        out_val = get_right_shape(other) - self.val
        out_grad = -self.grad
        return Variable(val=out_val, grad=out_grad)
 
    def __rtruediv__(self, other):
        """Implements division between other objects and Variables.
            See __div__ for reference.
        """
        new_val = get_right_shape(other)
        if abs(self.val) < 1e-4:
            raise ValueError("Divided by 0!")
        out_val = new_val / self.val
        out_grad = -new_val * self.grad / (self.val ** 2)
        return Variable(val=out_val, grad=out_grad)

    def __pow__(self, other):
        """Implements exponentiation between Variables and other objects,
            which are numeric values.
    
        INPUTS
        =======
        other: Float, or int.
            The power to which we are exponentiating our Variable.
    
        RETURNS
        ========
        Variable: A new variable whose val and grad are those resulting
            from the exponentiation of our Variable and other.
    
        EXAMPLES
        =========
        >>> x = Variable(2.0)
        >>> z = x ** 3
        >>> print(z)
        Value: [8.]
        Gradient: [12.]
        """
        new_val = get_right_shape(other)
        if self.val <= 0:
            raise ValueError("Power base cannot be smaller than 0!")
        out_val = self.val ** new_val
        out_grad = new_val * (self.val ** (new_val - 1)) * self.grad
        return Variable(val=out_val, grad=out_grad)

    def __rpow__(self, other):
        """Implements exponentiation between other objects, which are
            numeric values, and variables.
    
        INPUTS
        =======
        other: Float, or int.
            The base, which we are exponentiating to our Variable.
    
        RETURNS
        ========
        Variable: A new variable whose val and grad are those resulting
            from the exponentiation of other and Variable.
    
        EXAMPLES
        =========
        >>> x = Variable(2.0)
        >>> z = 3 ** x
        >>> print(z)
        Value: [9.]
        Gradient: [9.8875106]
        """
        new_val = get_right_shape(other)
        # Change later for vector variables
        if new_val <= 0:
            raise ValueError("Power base cannot be smaller than 0!")
        out_val = new_val ** self.val
        out_grad = np.log(new_val) * (new_val ** self.val) * self.grad
        return Variable(val=out_val, grad=out_grad) 
    
    def __neg__(self):
        """Implements negation (-1 * self) of Variable.
        """
        out_val = -self.val
        out_grad = -self.grad
        return Variable(val=out_val, grad=out_grad)
    
    def __eq__(self, other):
        #TODO-Compare valuees, gradients. 
        #Make sure it handles any value size, even one-dim. 
        return None
    
    def do_backward(self):
        #grad = np.eyes()
        #TODO-Clean the graph after the pass
        #TODO-Make it work on multi-dim data.
        grad = 1.
        for var in autodiff.config.reverse_graph:
            grad *= var.grad
        return grad
        

class ReverseVariable(Variable):
    """
    Overload __add__, __mul__  and so on. 
    """
    def __init__(self, *args, children=[]) :
        super().__init__(*args)
        self.children = children

    def __add__(self, other):
        if isinstance(other, ReverseVariable):
            out_val = self.val + other.val
            children = [self, other]
        else:
            out_val = self.val + other.val
            children = self
        return ReverseVariable(out_val, 1. , children=children) #1 rather than None-> None will init the grad with a matrix
    
    def do_backward(self):
        if len(self.children) == 0: #Root
            return self
        else:
            for child in self.children:
                child.grad *= self.grad #Chain rule
                print("Do BACWARD")
                child.do_backward() #Another loop ?


