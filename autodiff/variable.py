import numpy as np
from autodiff.utils import get_right_shape


class Variable:
    """ 
    Represents a variable, and carries the information flow
    within the computational graph. Under the current 
    implementation, variables are scalar valued.
    """
    def __init__(self, val, grad=1.): 
        """
        Variables are initialized with a value and a gradient.

         INPUTS
        =======
        val: float or int, required.
            Is the value of the variable.

        grad: float or int, optional. Default value is 1 (the seed).
            Is the gradient of the variable.

        EXAMPLE
        =========
        >>> x = Variable(2.0)
        """

        # Assure val and grad are correct shape (in preparation for
        # multivariate implementation)
        self.val = get_right_shape(val)
        #grad = np.ones((len(self.val), ))
        self.grad = get_right_shape(grad)
    
    def __repr__(self):
        """ When variables are printed, gives both val and grad"""
        return "Value: {}\nGradient: {}".format(self.val, self.grad)
    
    def __add__(self, other):
        """Implements addition (self + other) between Variables and
            other objects, which are either Variables or numeric values. 
    
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
        """Implements multiplication (self * other) between Variables and
            other objects, which are either Variables or numeric values.
    
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
        """Implements addition (other + self) between Variables and
            other objects. See __add__ for reference.
        """
        new_val = get_right_shape(other)
        out_val = self.val + new_val
        out_grad = self.grad
        return Variable(val=out_val, grad=out_grad)
    
    def __rmul__(self, other):
        """Implements multiplication (other * self) between Variables and
            other objects. See __mul__ for reference.
        """
        new_val = get_right_shape(other)
        out_val = self.val * new_val
        out_grad = self.grad * new_val 
        return Variable(val=out_val, grad=out_grad)

    def __sub__(self, other):
        """Implements subtraction (self - other) between Variables and
            other objects. See __add__ for reference.
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
        """Implements division (self / other) between Variables and
            other objects. See __mul__ for reference.
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
        """Implements subtraction (other - self) between Variables and
            other objects. See __sub__ for reference.
        """
        out_val = get_right_shape(other) - self.val
        out_grad = -self.grad
        return Variable(val=out_val, grad=out_grad)
 
    def __rtruediv__(self, other):
        """Implements division (other / self) between Variables and
            other objects. See __div__ for reference.
        """
        new_val = get_right_shape(other)
        if abs(self.val) < 1e-4:
                raise ValueError("Divided by 0!")
        out_val = new_val / self.val
        out_grad = -new_val * self.grad / (self.val ** 2)
        return Variable(val=out_val, grad=out_grad)

    def __pow__(self, other):
        """Implements exponentiation (self ^ other) between Variables and
            other objects, which are numeric values.
    
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
        out_val = self.val ** new_val
        out_grad = new_val * (self.val ** (new_val - 1)) * self.grad
        return Variable(val=out_val, grad=out_grad)

    def __rpow__(self, other):
        """Implements exponentiation (other ^ self) between Variables and
            other objects, which are numeric values.
    
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
            raise ValueError("power base cannot be smaller than 0!")
        out_val = new_val ** self.val
        out_grad = np.log(new_val) * (new_val ** self.val) * self.grad
        return Variable(val=out_val, grad=out_grad) 
    
    def __neg__(self):
        """Implements negation (-1 * self) of Variable.
        """
        out_val = -self.val
        out_grad = -self.grad
        return Variable(val=out_val, grad=out_grad)
    
    #def __getitem__(self):
        #TODO
        # return None
    #@classmethod
    #def _transform(cls, func):
    #    #TODO: assert type(func) == autodiff.Function:
    #    return cls()
