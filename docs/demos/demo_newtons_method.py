import numpy as np 
from autodiff.variable import Variable
import autodiff.function as F


def newtons_method(function, guess, epsilon):
    x = Variable(guess)
    f = function(x)
    i = 0
    max_out = False
    while abs(f.val) >= epsilon and max_out == False:
        x = x - f.val / f.grad
        f = function(x)
        print(x.val)
        i += 1
        if i >= 10000:
            max_out = True
            

def my_func(x):
    return 5*(x-2)**3

guess = 5
epsilon = 0.000001

newtons_method(my_func, guess, epsilon)