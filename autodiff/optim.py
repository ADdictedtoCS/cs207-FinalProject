import numpy as np 
import autodiff.function as F
from autodiff.variable import Variable
import matplotlib.pyplot as plt

class Optimizer:
    """
    init the optimizer with a learning rate and that kind of things (tol and so on).
    optimize a function fn, with respect to parameters.
    Inspired from torch where we give parameters ? 
    """
    def __init__(self, lr, tol, loss_fn, init_point):
        self.lr = lr
        self.tol = tol
        self.loss_fn = loss_fn
        self.current_point = Variable(init_point)

    def _step(self):
        #raise NotImplementedError
        return None
    
    def _eval(self, *args, **kwargs):
        """
        Output is a variable.
        """
        return self.loss_fn(*args, **kwargs)
    
    def minimize(self, nb_steps):
        it = 0 
        trajectory = []
        losses = []
        while it < nb_steps:
            loss = self._eval(self.current_point)
            self.current_point -= self.lr * loss.grad #By doing this, we directly create a new Variable
            trajectory.append(self.current_point.val)
            losses.append(loss.val)
            it +=1
        return losses, trajectory

if __name__ == "__main__":
    init_point = np.array([4,5])
    def my_loss_fn(X):
        x,y = F.unroll(X)
        if x.val < 0:
            x=-x #force x to be positive. Problem with derivative though
        else:
            pass
        #return 50-x*x-2*y*y  
        return x*y -y 
        
    gd = Optimizer(0.01, 0.001, my_loss_fn, init_point)
    l = gd._eval(gd.current_point)
    print(l)
    u =init_point - l.grad
    print(u)
    l,t = gd.minimize(1000)
    print('l',l,'\n') 
    print('t',t)
    plt.plot(l)
    plt.show()
    plt.plot(t)
    plt.show()

    



        
