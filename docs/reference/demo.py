import numpy as np 
from autodiff.variable import Variable
import autodiff.function as F

# Define a variable with an initial value
x = Variable(0.)
print("Input x", x)

# Define a function
def my_func(x):
    return F.sin(F.exp(x))

# Variable z is the result of calling function on x
z = my_func(x)

# Get value and gradient of z
print("Output z", z)

# Alternatively, with direct access to the value and gradient attributes.
print('The value is: {}'.format(z.val))
print('The gradient is: {}'.format(z.grad))