from autodiff.variable import Variable
from autodiff.variable import ReverseVariable as RVariable
import autodiff.function as F
import autodiff.optim as optim
import numpy as np
import pytest 

def my_loss_fn(X):
    x, y = X.unroll()
    return x*y

init_point = np.array([10,1])
gd = optim.Optimizer(0.01, 0.0001, my_loss_fn, init_point)

def test_not_implemented():
    init_point = np.array([4, 5])
    gd = optim.Optimizer(0.01, 0.0001, my_loss_fn, init_point)
    with pytest.raises(NotImplementedError):
        _,_,_  = gd.minimize(1000)

def test_type_error():
    init_point = True
    with pytest.raises(TypeError):
        gd = optim.Optimizer(0.01, 0.0001, my_loss_fn, init_point)
    init_point = RVariable([4,5])
    with pytest.raises(TypeError):
        gd = optim.Optimizer(0.01, 0.0001, my_loss_fn, init_point)

def test_assertion_error():
    init_point = np.array([4, 5])
    init_point = Variable(init_point)
    def loss_fn(X):
        return X+1
    with pytest.raises(AssertionError):
        gd = optim.Optimizer(0.01, 0.0001, loss_fn, init_point)
    with pytest.raises(AssertionError):
        gd = optim.Optimizer(-0.01, 0.0001, loss_fn, init_point)
    with pytest.raises(AssertionError):
        gd = optim.Optimizer(0.01, -0.0001, loss_fn, init_point)
    with pytest.raises(AssertionError):
        gd = optim.Optimizer(np.ndarray(2), 0.0001, loss_fn, init_point)
    with pytest.raises(AssertionError):
        gd = optim.Optimizer(2, np.array(0.0001), loss_fn, init_point)

def test_gd():
    init_point = np.array([10, 1])
    tol = 0.0001
    gd = optim.GradientDescent(0.1, tol, my_loss_fn, init_point)
    point, loss, trajectory = gd.minimize(1000)
    #gd.visualize(loss, trajectory)
    assert 0 <=loss[-1] < tol, "Convergence not reached"


def test_rms():
    init_point = np.array([10, 1])
    tol = 0.00001
    gd = optim.RMSProp(0.01, tol, my_loss_fn, init_point)
    point, loss, trajectory = gd.minimize(10000)
    #gd.visualize(loss, trajectory)
    assert loss[-1] < tol, "Convergence not reached"
    #assert loss[-1] >=tol , "Stopping because the loss is becoming negative"

    
test_not_implemented()
test_type_error()
test_assertion_error()
test_gd()
test_rms()
#test_visualize()




#init_point = np.array([4, 5])
#init_point = Variable(init_point)
#print(init_point.val, init_point.grad)
#z = init_point + 1
#print(z.val, z.val.shape, z.grad)

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


