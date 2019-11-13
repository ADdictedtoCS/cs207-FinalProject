import numpy as np 
from variable import Variable
import function as F

def my_func(x):
    return F.sin(F.exp(x))

Functions_available = [F.exp, F.sin]
x = Variable(np.array([0]))
print(x)
y = F.exp(x)
z = F.sin(y)
print(y)
print(z)

Z = my_func(x)
print("MY FUNC", Z)
