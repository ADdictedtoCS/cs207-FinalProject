from autodiff.variable import Variable
from autodiff.variable import ReverseVariable as RVariable
import autodiff.function as F
import autodiff.optim as optim
import numpy as np

def my_loss_fn(X):
    x, y = X.unroll()

init_point = np.array([4,5])
gd = optim.Optimizer(0.01, 0.0001, my_loss_fn, init_point)

def test_not_implemented():
    with pytest.Raises(NotImplementedError):
        _,_,_  = gd.minimize(1000)
    
#def init_error():
#    with pytest.Raises(NotImplementedError):

"""
if __name__ == "__main__":
    init_point = np.array([4,5])
    try: 
        import matplotlib.pyplot as plt 
    except:
        print("Please import matplotlib module if you want visualization.")
    
    def my_loss_fn(X):
        x,y = X.unroll()
        #if x.val < 0:
        #    x=-x #force x to be positive. Problem with derivative though
        #else:
        #    pass
        #return 50-x*x-2*y*y  
        return x*y -y 

    gd = Optimizer(0.01, 0.0001, my_loss_fn, init_point)
    l = gd._eval(gd.current_point)
    print(l)
    u =init_point - l.grad
    print(u)
    try:
        a_gd,l_gd,t_gd = gd.minimize(1000)
    except Exception as e:
        print(e)
        print('Got into the exception')
    gd = GradientDescent(0.01,-10e8, my_loss_fn, init_point)
    a_gd, l_gd, t_gd = gd.minimize(10000)
    #print('l',l,'\n') 
    #print('t',t)
    #plt.plot(l)
    #plt.show()
    #plt.plot(t)
    #plt.show()
    prop = RMSProp(0.01, -10e8, my_loss_fn, init_point, beta=0.9)
    print(prop)
    a, l, t = prop.minimize(10000)
    plt.plot(l)
    #plt.plot(l_gd, 'r')
    #plt.plot(t)
    plt.show()
"""


