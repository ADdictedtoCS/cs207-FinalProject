import numpy as np 
from autodiff.variable import Variable
import autodiff.function as F

def my_func(x):
    return F.sin(F.exp(x))

x = Variable(0.)
print("Input x", x)

z = my_func(x)
print("Output z", z)