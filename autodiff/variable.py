import numpy as np
from autodiff.utils import get_right_shape
import autodiff

class Variable:
    """ 
    A Variable is an object, which carries the information flow
    within the computational graph.
    """
    def __hash__(self):
        return id(self)

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
        self.grad = {}
        #grad = np.ones((len(self.val), ))
        #We now assume that grad is a n-dimensional element, where n=len(val).
        if grad is None: #if created from scratch.
        #    self.grad = {self: np.eye(self.val.shape[0])}
            if isinstance(self.val, float):
                self.grad[self] = 1
            else:
                self.grad[self] = np.eye(self.val.shape[0])
        else:
            #self.grad = get_right_shape(grad) 
            self.grad = grad #If not created from scratch, assumes we already hav a gradient under the right form. 
        #print('Initializing the gradient with {}'.format(self.grad))
    
    def __repr__(self):
        """ When variables are printed, gives both val and grad"""
        return "Value: {}\nGradient: {}".format(self.val, self.grad)

    def merge_grad(self, function, dict1, dict2):
        res = {}
        for var in dict1:
            if var in dict2:
                res[var] = function(dict1[var], dict2[var])
                # print(dict1[var], dict2[var], res[var])
            else:
                res[var] = function(dict1[var], 0)
                # print(dict1[var], res[var])
        for var in dict2:
            if var not in res:
                res[var] = function(0, dict2[var])
                # print(dict2[var], res[var])
        return res

    def single_grad(self, function, dict1):
        res = {}
        for var in dict1:
            res[var] = function(dict1[var])
        return res
    
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
            def _add(a, b):
                return a + b
            out_grad = self.merge_grad(_add, self.grad, other.grad)
            # out_grad = self.grad + other.grad
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
            def _mul(x, y):
                return x * other.val + self.val * y
            out_grad = self.merge_grad(_mul, self.grad, other.grad)
            # out_grad = self.grad * other.val + self.val * other.grad
            return Variable(val=out_val, grad=out_grad)
        else:
            new_val = get_right_shape(other)
            out_val = self.val * new_val
            def _mul(a):
                return a * new_val
            out_grad = self.single_grad(_mul, self.grad)
            # out_grad = self.grad * new_val
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
        out_val = new_val * self.val
        def _mul(a):
            return new_val * a
        out_grad = self.single_grad(_mul, self.grad)
        # out_grad = self.grad * new_val 
        return Variable(val=out_val, grad=out_grad)

    def __sub__(self, other):
        """Implements subtraction between Variables and other objects.
            See __add__ for reference.
        """
        if isinstance(other, Variable):
            out_val = self.val - other.val
            def _sub(a, b):
                return a - b
            out_grad = self.merge_grad(_sub, self.grad, other.grad)
            # out_grad = self.grad - other.grad
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
            if not isinstance(other.val, float):
                raise ValueError("Vector cannot be the denominator")
            if abs(other.val) < 1e-4:
                raise ValueError("Divided by 0!") 
            out_val = self.val / other.val
            def _div(a, b):
                return (a * other.val - self.val * b) / (other.val ** 2)
            out_grad = self.merge_grad(_div, self.grad, other.grad)
            return Variable(val=out_val, grad=out_grad)
        else: 
            new_val = get_right_shape(other)
            if not isinstance(other, float):
                raise ValueError("Vector cannot be the denominator")
            if abs(new_val) < 1e-4:
                raise ValueError("Divided by 0!")
            out_val = self.val / new_val
            def _div(a):
                return a / new_val
            out_grad = self.single_grad(_div, self.grad)
            return Variable(val=out_val, grad=out_grad)

    def __rsub__(self, other):
        """Implements subtraction between other objects and Variables.
            See __sub__ for reference.
        """
        out_val = get_right_shape(other) - self.val
        def _neg(a):
            return -a
        out_grad = self.single_grad(_neg, self.grad)
        # out_grad = -self.grad
        return Variable(val=out_val, grad=out_grad)
 
    def __rtruediv__(self, other):
        """Implements division between other objects and Variables.
            See __div__ for reference.
        """
        new_val = get_right_shape(other)
        if not isinstance(self.val, float):
            raise ValueError("Vector cannot be the denominator")
        if abs(self.val) < 1e-4:
            raise ValueError("Divided by 0!")
        out_val = new_val / self.val
        def _div(a):
            return -new_val * a / (self.val ** 2)
        out_grad = self.single_grad(_div, self.grad)
        # out_grad = -new_val * self.grad / (self.val ** 2)
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
        if not isinstance(new_val, float):
            raise ValueError("Exponent cannot be a vector!")
        if isinstance(self.val, float):
            # if self.val <= 0:
                # raise ValueError("Power base cannot be smaller than 0!")
            out_val = self.val ** new_val
            def _pow(a):
                return new_val * (self.val ** (new_val - 1)) * a
            out_grad = self.single_grad(_pow, self.grad)
            # out_grad = new_val * (self.val ** (new_val - 1)) * self.grad
        else:
            out_val = []
            for i in range(self.val.shape[0]):
                out_val.append(self.val[i, 0] ** new_val)
            # out_val = [val ** new_val for val in self.val]
            out_grad = {}
            for var in self.grad:
                height = self.grad[var].shape[0]
                width = self.grad[var].shape[1]
                o_grad = np.zeros(self.grad[var].shape)
                for i in range(height):
                    for j in range(width):
                        o_grad[i, j] = new_val * (self.val[i, 0] ** (new_val - 1)) * self.grad[var][i, j]
                out_grad[var] = o_grad
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
        # if new_val <= 0:
            # raise ValueError("Power base cannot be smaller than 0!")
        if not isinstance(self.val, float):
            raise ValueError("Exponent canont be a vector!")
        out_val = new_val ** self.val
        def _pow(a):
            return np.log(new_val) * (new_val ** self.val) * a
        out_grad = self.single_grad(_pow, self.grad)
        # out_grad = np.log(new_val) * (new_val ** self.val) * self.grad
        return Variable(val=out_val, grad=out_grad) 
    
    def __neg__(self):
        """Implements negation (-1 * self) of Variable.
        """
        out_val = -self.val
        def _neg(a):
            return -a
        out_grad = self.single_grad(_neg, self.grad)
        # out_grad = -self.grad
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
        

class ReverseVariable():
    """
    Overload __add__, __mul__  and so on. 
    """
    def __init__(self, val) :
        # super().__init__(*args)
        self.children = []
        self.val = val
        self.grad = None
        self.left = None
        self.leftgrad = None
        self.right = None
        self.rightgrad = None

    def __add__(self, other):
        if isinstance(other, ReverseVariable):
            out_val = self.val + other.val
            res = ReverseVariable(out_val)
            self.children.append(res)
            other.children.append(res)
            res.left = self
            res.leftgrad = 1
            res.right = other
            res.rightgrad = 1
            return res
            # out_grad = get_right_shape([1., 1.])
            # children = [self, other]
            # return ReverseVariable(out_val, out_grad, children=children)
        else:
            out_val = self.val + other
            res = ReverseVariable(out_val)
            self.children.append(res)
            res.left = self
            res.leftgrad = 1
            return res
            # out_grad = self.grad
            # children = self
        # return ReverseVariable(out_val, out_grad , children=children) #1 rather than None-> None will init the grad with a matrix

    def __radd__(self, other):
        out_val = self.val + other
        res = ReverseVariable(out_val)
        self.children.append(res)
        res.left = self
        res.leftgrad = 1
        return res

    def __sub__(self, other):
        if isinstance(other, ReverseVariable):
            out_val = self.val - other.val
            res = ReverseVariable(out_val)
            self.children.append(res)
            other.children.append(res)
            res.left = self
            res.leftgrad = 1
            res.right = other
            res.rightgrad = -1
            return res
        else:
            out_val = self.val - other
            res = ReverseVariable(out_val)
            self.children.append(res)
            res.left = self
            res.leftgrad = 1
            return res

    def __rsub__(self, other):
        out_val = other - self.val
        res = ReverseVariable(out_val)
        self.children.append(res)
        res.left = self
        res.leftgrad = -1
        return res

    def __mul__(self, other):
        if isinstance(other, ReverseVariable):
            out_val = self.val * other.val
            res = ReverseVariable(out_val)
            self.children.append(res)
            other.children.append(res)
            res.left = self
            res.leftgrad = other.val
            res.right = other
            res.rightgrad = self.val
            return res
            # out_grad = get_right_shape([other.val, self.val])
            # children = [self, other]
        else:
            out_val = self.val * other
            res = ReverseVariable(out_val)
            self.children.append(res)
            res.left = self
            res.leftgrad = other
            return res
            #We need a two-dimensional grad that controls the bakcward flow.
            #out_grad = get_right_shape( [1., 1.])
            # out_grad = self.grad * other
            # children = self
        # return ReverseVariable(out_val, out_grad, children=children)

    def __rmul__(self, other):
        out_val = self.val * other
        res = ReverseVariable(out_val)
        self.children.append(res)
        res.left = self
        res.leftgrad = other
        return res

    def __truediv__(self, other):
        if isinstance(other, ReverseVariable):
            out_val = self.val / other.val
            res = ReverseVariable(out_val)
            self.children.append(res)
            other.children.append(res)
            res.left = self
            res.leftgrad = 1.0 / other.val
            res.right = other
            res.rightgrad = -self.val / (other.val ** 2)
            return res
        else:
            out_val = self.val / other
            res = ReverseVariable(out_val)
            self.children.append(res)
            res.left = self
            res.leftgrad = 1.0 / other
            return res

    def __rtruediv__(self, other):
        out_val = other / self.val
        res = ReverseVariable(out_val)
        self.children.append(res)
        res.left = self
        res.leftgrad = -other / (self.val ** 2)
        return res

    def __pow__(self, other):
        new_val = get_right_shape(other)
        if self.val <= 0:
            raise ValueError("Power base cannot be smaller than 0!")
        if self.val.shape[0] == 1:
            out_val = self.val ** new_val
            res = ReverseVariable(out_val)
            self.children.append(res)
            res.left = self
            res.leftgrad = new_val * (self.val ** (new_val - 1))
            return res
        else:
            out_val = [val ** new_val for val in self.val]
            res = ReverseVariable(out_val)
            self.children.append(res)
            res.left = self
            res.leftgrad = np.zeros((self.val.shape[0], self.val.shape[0]))
            for i in range(self.val.shape[0]):
                res.leftgrad[i, i] = new_val * (self.val[i, 0] ** (new_val - 1))
            return res

    def __rpow__(self, other):
        new_val = get_right_shape(other)
        if new_val <= 0:
            raise ValueError("Power base cannot be smaller than 0!")
        if self.val.shape[0] > 1:
            raise ValueError("The exponent canont be a multi-dimension vector!")
        out_val = new_val ** self.val
        res = ReverseVariable(out_val)
        self.children.append(res)
        res.left = self
        res.leftgrad = np.log(new_val) * (new_val ** self.val)
        return res

    def __neg__(self):
        out_val = -self.val
        res = ReverseVariable(out_val)
        self.children.append(res)
        res.left = self
        res.leftgrad = -1
        return res

    def check_children(self):
        for child in self.children:
            if child.grad == None:
                return False
        return True

    def reverse(self):
        if self.grad != None or not self.check_children():
            return
        sum = 0
        for child in self.children:
            if child.left == self:
                sum += child.grad * child.leftgrad
            elif child.right == self:
                sum += child.grad * child.rightgrad
        self.grad = sum
        if self.left != None:
            self.left.reverse()
        if self.right != None:
            self.right.reverse()
    
    def do_backward(self):
        #TODO-Cleaning the graph.
        if len(self.children) == 0: #Root
            return self
        else:
            for child, depart_grad in zip(self.children, self.grad): #Two childrens means two grad. coord
                child.grad *= depart_grad #Chain rule.
                print("Do BACKWARD")
                child.do_backward() #Another loop ?


