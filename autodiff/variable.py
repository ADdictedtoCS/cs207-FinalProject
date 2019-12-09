import numpy as np
from autodiff.utils import get_right_shape, close
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
        >>> x.val
        2.0
        >>> x = Variable((2))
        >>> x.val
        2.0
        >>> x = Variable(np.array([2, 3]))
        >>> x.val
        array([[2.],
               [3.]])
        """

        # Assure val and grad are correct shape (in preparation for
        # multivariate implementation)
        self.val = get_right_shape(val)
        #We now assume that grad is a n-dimensional element, where n=len(val).
        if grad is None: #if created from scratch.
            if isinstance(self.val, float):
                self.grad = 1.0
            else:
                self.grad = np.eye(self.val.shape[0])
        else:
            #If not created from scratch, assumes we already have a gradient under the right form.
            self.grad = get_right_shape(grad)
            
    def __repr__(self):
        """ When variables are printed, gives both val and grad"""
        return "Value: {}\nGradient: {}".format(self.val, self.grad)
   
    def unroll(self, unroll_list=None):
        """
        Unroll variable vector so that you can use part of the variable vector

        INPUTS
        =======
        unroll_list: how do you want to unroll the vector variable? Unroll list is defining the dimension of each sub-variable.
        For example [1, 2, 3], will divide the 6-d vector to 3 sub-variables,
        with dimension of 1, 2, and 3, correspondingly.
        The sum of the dimensions should be equal to the vector dimension.

        OUTPUTS
        =======
        A list of sub-variables.
        For example, with unroll list of [1, 2, 3], it will return a list of 3 variables
        With dimension 1, 2, and 3 for each variable. 
        """
        if unroll_list == None:
            if isinstance(self.val, float):
                return [self]
            else:
                var_list = []
                for i in range(self.val.shape[0]):
                    out_val = self.val[i, 0]
                    out_grad = np.ndarray((1, self.val.shape[0]), dtype=float)
                    for j in range(self.val.shape[0]):
                        out_grad[0, j] = self.grad[i, j]
                    var_list.append(Variable(val=out_val, grad=out_grad))
                return var_list
        else:
            if not isinstance(unroll_list, list):
                raise TypeError("Please unroll with a list!")
            for i in unroll_list:
                if not isinstance(i, int) or i <= 0:
                    raise ValueError("Unroll list should be positive numbers!")
            s = sum(unroll_list)
            if isinstance(self.val, float):
                if s != 1:
                    raise ValueError("Cannot unroll!")
                return [self]
            else:
                if s != self.val.shape[0]:
                    raise ValueError("Cannot unroll!")
                var_list = []
                s = 0
                for dim in unroll_list:
                    out_val = self.val[s:s+dim]
                    if dim == 1:
                        out_val = self.val[s, 0]
                    out_grad = np.ndarray((dim, self.val.shape[0]), dtype=float)
                    for i in range(out_grad.shape[0]):
                        for j in range(out_grad.shape[1]):
                            out_grad[i, j] = self.grad[s+i, j]
                    var_list.append(Variable(val=out_val, grad=out_grad))
                    s += dim
                return var_list
                        
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
        >>> X = Variable([2, 4])
        >>> var_list = X.unroll()
        >>> x = var_list[0]
        >>> y = var_list[1]
        >>> z = x + y
        >>> z.val
        6.0
        >>> z.grad
        array([[1., 1.]])
        """
        if isinstance(other, Variable):
            out_val = self.val + other.val
            # def _add(a, b):
            #     return a + b
            # out_grad = self.merge_grad(_add, self.grad, other.grad)
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
        other: Variable, float, int, or vectors.
            The object with which we are multiplying our Variable.
            It will be either a vector times a scalar, or a scalar times a vector
            If you want matrix multiplication, please use Dot function
    
        RETURNS
        ========
        Variable: A new variable whose val and grad are those resulting
            from the multiplication of our Variable and other.
    
        EXAMPLES
        =========
        >>> X = Variable([2, 4])
        >>> var_list = X.unroll()
        >>> x = var_list[0]
        >>> y = var_list[1]
        >>> z = x * y
        >>> z.val
        8.0
        >>> z.grad
        array([[4., 2.]])
        """
        if isinstance(other, Variable):
            out_val = np.dot(self.val, other.val)
            # def _mul(x, y):
                # return x * other.val + self.val * y
            # out_grad = self.merge_grad(_mul, self.grad, other.grad)
            out_grad = np.dot(self.grad, other.val) + np.dot(self.val, other.grad)
            return Variable(val=out_val, grad=out_grad)
        else:
            new_val = get_right_shape(other)
            out_val = np.dot(self.val, new_val)
            # def _mul(a):
                # return a * new_val
            # out_grad = self.single_grad(_mul, self.grad)
            out_grad = np.dot(self.grad, new_val)
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
        out_val = np.dot(new_val, self.val)
        # def _mul(a):
            # return new_val * a
        # out_grad = self.single_grad(_mul, self.grad)
        out_grad = np.dot(self.grad, new_val) 
        return Variable(val=out_val, grad=out_grad)

    def __sub__(self, other):
        """Implements subtraction between Variables and other objects.
            See __add__ for reference.
        """
        if isinstance(other, Variable):
            out_val = self.val - other.val
            # def _sub(a, b):
                # return a - b
            # out_grad = self.merge_grad(_sub, self.grad, other.grad)
            out_grad = self.grad - other.grad
            return Variable(val=out_val, grad=out_grad)
        else:
            new_val = get_right_shape(other)
            out_val = self.val - new_val
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
            # def _div(a, b):
                # return (a * other.val - self.val * b) / (other.val ** 2)
            # out_grad = self.merge_grad(_div, self.grad, other.grad)
            out_grad = (np.dot(self.grad, other.val) - np.dot(self.val, other.grad)) / (other.val ** 2)
            return Variable(val=out_val, grad=out_grad)
        else: 
            new_val = get_right_shape(other)
            if not isinstance(new_val, float):
                raise ValueError("Vector cannot be the denominator")
            if abs(new_val) < 1e-4:
                raise ValueError("Divided by 0!")
            out_val = self.val / new_val
            # def _div(a):
                # return a / new_val
            # out_grad = self.single_grad(_div, self.grad)
            out_grad = self.grad / new_val
            return Variable(val=out_val, grad=out_grad)

    def __rsub__(self, other):
        """Implements subtraction between other objects and Variables.
            See __sub__ for reference.
        """
        out_val = get_right_shape(other) - self.val
        # def _neg(a):
            # return -a
        # out_grad = self.single_grad(_neg, self.grad)
        out_grad = -self.grad
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
        # def _div(a):
            # return -new_val * a / (self.val ** 2)
        # out_grad = self.single_grad(_div, self.grad)
        out_grad = -np.dot(new_val, self.grad) / (self.val ** 2)
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
        >>> z.val
        8.0
        >>> z.grad
        12.0
        """
        if isinstance(other, Variable):
            if not isinstance(other.val, float):
                raise ValueError("Exponent cannot be a vector!")
            if not isinstance(self.val, float):
                raise ValueError("Base as a vector not supported!")
            out_val = self.val ** other.val
            out_grad = np.dot(out_val, (np.dot(other.grad, np.log(self.val)) + np.dot(other.val / self.val, self.grad)))
            return Variable(val=out_val, grad=out_grad)
        else:
            new_val = get_right_shape(other)
            if not isinstance(new_val, float):
                raise ValueError("Exponent cannot be a vector!")
            if isinstance(self.val, float):
                out_val = self.val ** new_val
                out_grad = np.dot(np.dot(new_val, (self.val ** (new_val - 1))), self.grad)
            else:
                out_val = []
                for i in range(self.val.shape[0]):
                    out_val.append(self.val[i, 0] ** new_val)
                out_val = get_right_shape(out_val)
                height = self.grad.shape[0]
                width = self.grad.shape[1]
                o_grad = np.zeros(self.grad.shape)
                for i in range(height):
                    for j in range(width):
                        o_grad[i, j] = np.dot(np.dot(new_val, (self.val[i, 0] ** (new_val - 1))), self.grad[i, j])
                out_grad = o_grad
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
        >>> z.val
        9.0
        >>> z.grad
        9.887510598012987
        """
        new_val = get_right_shape(other)
        # Change later for vector variables
        # if new_val <= 0:
            # raise ValueError("Power base cannot be smaller than 0!")
        if not isinstance(self.val, float):
            raise ValueError("Exponent canont be a vector!")
        out_val = new_val ** self.val
        # def _pow(a):
            # return np.log(new_val) * (new_val ** self.val) * a
        # out_grad = self.single_grad(_pow, self.grad)
        out_grad = np.dot(np.dot(np.log(new_val), (new_val ** self.val)), self.grad)
        return Variable(val=out_val, grad=out_grad) 
    
    def __neg__(self):
        """Implements negation (-1 * self) of Variable.
        """
        out_val = -self.val
        # def _neg(a):
            # return -a
        # out_grad = self.single_grad(_neg, self.grad)
        out_grad = -self.grad
        return Variable(val=out_val, grad=out_grad)
    
    def __eq__(self, other):
        #TODO-DOC about this
        if isinstance(other, Variable):
            if close(self.val, other.val) and close(self.grad, other.grad):
                return True
            else:
                return False
        else:
            new_val = get_right_shape(other)
            if close(self.val, new_val) and close(self.grad, get_right_shape(np.zeros(self.grad.shape))):
                return True
            else:
                return False

    # def __req__(self, other):
    #     new_val = get_right_shape(other)
    #     if close(self.val, new_val) and close(self.grad, get_right_shape(np.zeros(self.grad.shape))):
    #         return True
    #     else:
    #         return False

    def __ne__(self, other):
        return not self.__eq__(other)

        

class ReverseVariable():
    
    def __init__(self, val) :
        """
        ReverseVariables are initialized with a value.

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
        >>> x = ReverseVariable(2.0)
        >>> x.val
        2.0
        >>> x = ReverseVariable((2))
        >>> x.val
        2.0
        >>> x = ReverseVariable(np.array([2, 3]))
        >>> x.val
        array([[2.],
               [3.]])
        """

        self.children = []
        self.val = get_right_shape(val)
        self.grad = None
        self.left = None
        self.leftgrad = None
        self.right = None
        self.rightgrad = None
        self.tag = 0

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
        >>> x = ReverseVariable(4)
        >>> y = ReverseVariable(2)
        >>> z = x + y
        >>> z.val
        6.0
        >>> z.reverse()
        >>> x.grad
        1.0
        >>> y.grad
        1.0
        """
        if isinstance(other, ReverseVariable):
            out_val = self.val + other.val
            res = ReverseVariable(out_val)
            self.children.append(res)
            other.children.append(res)
            res.left = self
            res.leftgrad = 1.0
            res.right = other
            res.rightgrad = 1.0
            return res
            # out_grad = get_right_shape([1., 1.])
            # children = [self, other]
            # return ReverseVariable(out_val, out_grad, children=children)
        else:
            out_val = self.val + get_right_shape(other)
            res = ReverseVariable(out_val)
            self.children.append(res)
            res.left = self
            res.leftgrad = 1.0
            return res
            # out_grad = self.grad
            # children = self
        # return ReverseVariable(out_val, out_grad , children=children) #1 rather than None-> None will init the grad with a matrix

    def __radd__(self, other):
        """Implements addition between other objects and Variables.
            See __add__ for reference.
        """
        out_val = self.val + get_right_shape(other)
        res = ReverseVariable(out_val)
        self.children.append(res)
        res.left = self
        res.leftgrad = 1
        return res

    def __sub__(self, other):
        """Implements substraction between Variables.
            See __add__ for reference.
        """
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
            out_val = self.val - get_right_shape(other)
            res = ReverseVariable(out_val)
            self.children.append(res)
            res.left = self
            res.leftgrad = 1
            return res

    def __rsub__(self, other):
        """Implements substraction between other objects and Variables.
            See __add__ for reference.
        """
        out_val = get_right_shape(other) - self.val
        res = ReverseVariable(out_val)
        self.children.append(res)
        res.left = self
        res.leftgrad = -1
        return res

    def __mul__(self, other):
        """Implements multiplication between Variables.
            See __add__ for reference.
        """
        if isinstance(other, ReverseVariable):
            out_val = np.dot(self.val, other.val)
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
            out_val = np.dot(self.val, get_right_shape(other))
            res = ReverseVariable(out_val)
            self.children.append(res)
            res.left = self
            res.leftgrad = get_right_shape(other)
            return res
            #We need a two-dimensional grad that controls the bakcward flow.
            #out_grad = get_right_shape( [1., 1.])
            # out_grad = self.grad * other
            # children = self
        # return ReverseVariable(out_val, out_grad, children=children)

    def __rmul__(self, other):
        """Implements multiplication between other objects and Variables.
            See __add__ for reference.
        """
        out_val = np.dot(self.val, get_right_shape(other))
        res = ReverseVariable(out_val)
        self.children.append(res)
        res.left = self
        res.leftgrad = get_right_shape(other)
        return res

    def __truediv__(self, other):
        """Implements division between Variables.
            See __add__ for reference.
        """
        if isinstance(other, ReverseVariable):
            if not isinstance(other.val, float):
                raise ValueError("Vector cannot be the denominator!")
            if close(other.val, 0):
                raise ValueError("Divided by 0!")
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
            new_val = get_right_shape(other)
            if not isinstance(new_val, float):
                raise ValueError("Vector cannot be the denominator!")
            if close(new_val, 0):
                raise ValueError("Divided by 0!")
            out_val = self.val / new_val
            res = ReverseVariable(out_val)
            self.children.append(res)
            res.left = self
            res.leftgrad = 1.0 / new_val
            return res

    def __rtruediv__(self, other):
        """Implements division between other objects and Variables.
            See __add__ for reference.
        """
        new_val = get_right_shape(other)
        if not isinstance(self.val, float):
            raise ValueError("Vector cannot be the denominator!")
        if close(self.val, 0):
                raise ValueError("Divided by 0!")
        out_val = new_val / self.val
        res = ReverseVariable(out_val)
        self.children.append(res)
        res.left = self
        res.leftgrad = -new_val / (self.val ** 2)
        return res

    def __pow__(self, other):
        """Implements power between Variables.
            See __add__ for reference.
        """
        if isinstance(other, ReverseVariable):
            if not isinstance(other.val, float):
                raise ValueError("Exponent not a number")
            if not isinstance(self.val, float):
                raise ValueError("Base a vector not supported")
            out_val = self.val ** other.val
            res = ReverseVariable(out_val)
            self.children.append(res)
            other.children.append(res)
            res.left = self
            res.leftgrad = np.dot(other.val, (self.val ** (other.val - 1)))
            res.right = other
            res.rightgrad = np.dot(np.log(other.val), self.val ** other.val)
            return res
        else:
            new_val = get_right_shape(other)
            if not isinstance(new_val, float):
                raise ValueError("Exponent not a number")
            # if self.val <= 0:
                # raise ValueError("Power base cannot be smaller than 0!")
            if isinstance(self.val, float):
                out_val = self.val ** new_val
                res = ReverseVariable(out_val)
                self.children.append(res)
                res.left = self
                res.leftgrad = np.dot(new_val, (self.val ** (new_val - 1)))
                return res
            else:
                out_val = [val ** new_val for val in self.val]
                res = ReverseVariable(out_val)
                self.children.append(res)
                res.left = self
                res.leftgrad = np.zeros((self.val.shape[0], self.val.shape[0]))
                for i in range(self.val.shape[0]):
                    res.leftgrad[i, i] = np.dot(new_val, (self.val[i, 0] ** (new_val - 1)))
                return res

    def __rpow__(self, other):
        """Implements power between other objects and Variables.
            See __add__ for reference.
        """
        new_val = get_right_shape(other)
        # if new_val <= 0:
            # raise ValueError("Power base cannot be smaller than 0!")
        if not isinstance(self.val, float):
            raise ValueError("The exponent canont be a multi-dimension vector!")
        out_val = new_val ** self.val
        res = ReverseVariable(out_val)
        self.children.append(res)
        res.left = self
        res.leftgrad = np.dot(np.log(new_val), (new_val ** self.val))
        return res

    def __neg__(self):
        """Implements negation for Variables.
            See __add__ for reference.
        """
        out_val = -self.val
        res = ReverseVariable(out_val)
        self.children.append(res)
        res.left = self
        res.leftgrad = -1
        return res

    def check_children(self, tag):
        for child in self.children:
            if child.grad is None and child.tag == tag:
                return False
        return True

    def reverse(self):
        """Start the reverse pass for the ReverseVariable.
        After running this method, the grad for all ReverseVariable is calculated.
        Assume you are calling f.reverse(), and you want df/dx, 
        then you can get it using x.grad.
        """
        self.clean_grad(id(self))
        if isinstance(self.val, float):
            self.grad = 1.0
        else:
            self.grad = np.eye(self.val.shape[0])
        if self.left is not None:
            self.left._reverse(id(self))
        if self.right is not None:
            self.right._reverse(id(self))

    def _reverse(self, tag):
        if self.grad is not None or not self.check_children(tag):
            return
        sum = 0.0
        lastchild = None
        for child in self.children:
            if child == lastchild:
                sum += np.dot(child.grad, child.rightgrad)
                lastchild = child
                continue
            if child.tag == tag:
                if child.left == self:
                    sum += np.dot(child.grad, child.leftgrad)
                elif child.right == self:
                    sum += np.dot(child.grad, child.rightgrad)
            lastchild = child
        self.grad = sum
        if self.left is not None:
            self.left._reverse(tag)
        if self.right is not None:
            self.right._reverse(tag)

    def clean_grad(self, tag):
        self.grad = None
        self.tag = tag
        if self.left is not None:
            self.left.clean_grad(tag)
        if self.right is not None:
            self.right.clean_grad(tag)
