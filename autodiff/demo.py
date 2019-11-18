import numpy as np 
from autodiff.variable import Variable
import autodiff.function as F

def my_func(x):
    return F.sin(F.exp(x))

def my_func2(x):
    return F.exp(x)

#Functions_available = [F.exp, F.sin]
# x=0, y=3, z=5
#X = Variable(np.array([0, 3, 5]))
#Y = Variable(np.array([5,7,8])
##Z = [[0, 3, 5, 5, 7, 8]]
#def func(X, Y):
#return F.exp(X) 

#F(x) equivalent F.__call__(x)

x = Variable(0.)
print("Input x", x)

z = my_func2(x)
print("Output z", z)

