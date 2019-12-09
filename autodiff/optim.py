import numpy as np 
import autodiff.function as F
from autodiff.variable import Variable
from autodiff.variable import ReverseVariable
#import matplotlib.pyplot as plt

class Optimizer:
    """
    Optimizer with 
    optimize a function fn, with respect to parameters.
    Inspired from torch where we give parameters directly ? 
    - If fwd mode, loss.grad returns the grad. All good.
    - If bkwd mode--> we need to do make the reverse directly. 
    - All of this is supposed to be mentioned. 
    """
    def __init__(self, lr, tol, loss_fn, init_point):
        self.lr = lr
        self.tol = tol
        self.loss_fn = loss_fn
        try:
            self.current_point = Variable(init_point)
        except TypeError as e:
            if isinstance(Variable, init_point):
                self.current_point = init_point
            else:
                raise TypeError(e)

    def _step(self, *args, **kwargs):
        raise NotImplementedError

    def _eval(self, *args, **kwargs):
        """
        Output is a variable.
        Keep args if ever we want variable length inputs
        """
        return self.loss_fn(*args, **kwargs)
    
    def minimize(self, nb_steps, keep_track=True):
        """
        Keep track is a bool-> True returns the different losses/points obtained durring optim.
        """
        trajectory = []
        losses = []
        it = 0 
        loss = Variable(val=self.tol+1) #Randomly initialize the loss to get into the 
        while it < nb_steps and loss.val > self.tol:
            loss = self._eval(self.current_point)
            #self.current_point -= self.lr * loss.grad #By doing this, we directly create a new Variable
            self._step(loss)
            #keep track of our thing
            trajectory.append(self.current_point.val)
            losses.append(loss.val)
            it +=1
        print('Minimized the function for {} steps.'.format(it))
        if keep_track:
            return self.current_point, losses, trajectory
        else:
            return self.current_point

    def __repr__(self):
        return str(vars(self))

class GradientDescent(Optimizer):
    def _step(self, loss):
        """
        Assumes loss has a grad attribute.
        """
        self.current_point -= self.lr * loss.grad
        #try: #RVariable
        #    #The reverse mode works differently from the fwd. This is handled directly. 
        #    loss.reverse()
        #    self.current_point -= self.lr * loss.grad
        #except: 
            #self.current_point -= self.lr * loss.grad
        
class RMSProp(Optimizer):
    """
    #TODO-Add citation and explanations and so on. 
    """
    def __init__(self, *args, beta=0.9):
        super().__init__(*args)
        self.beta = beta

    def _step(self, loss, eps=10e-6):
        try:
            self.avg = self.beta * self.avg + (1 - self.beta) * loss.grad ** 2 #Loss val and grad should be (N,) and (N,1)
        except Exception as e:#self.avg does not exist yet. Needs to create it. 
            print(e)
            self.avg = np.zeros(loss.grad.shape, dtype=np.float64)
            self.avg = self.beta * self.avg + (1 - self.beta) * loss.grad ** 2
        #Update rule
        self.current_point -= self.lr * loss.grad / (np.sqrt(self.avg) + eps)#Element wise sqrt. Add eps for numerical overflow. 
    
#class Adam(Optimizer):
#    def __init__(self, *args, beta1=0.9, beta2=0.99):
#        super().__init__(*args)
#        self.beta1 = beta1
#        self.beta2 = beta2

#    def _step(self, loss):
#        return NotImplementedError




    



        
