## Some comments-Théo

General remark: When an user defines the function. 
def my_func(X:[x,y]):
    return sth(x,y)
We can not add input dimensions or change of variables such as z = x+y -> et the der. wrt to z. 

#TODO modify the utils-> reshape array. Not up to date anymore. 

### Add an other argument, to say which are the variables and the non variables in the model ? 
Sth like requires_grad ? -> This is optional though. 
E.G def my_func(X, alpha, beta, gamma):
    X1 = F.dot(X, alpha)

### A more general unroll ? 
For instance, X = Variable([1,2,3])-> x, Z = F.unroll(X)-> Z is dimension 2. 
-> The grad has to be shape 2. This would imply that the dimensions of the grad would be changed within 
a function. 

#TODO-> Garbage management ? We are creating variables and that's a bunch of objects we are creating. 

#TODO for backward: create a backward.py that holds the graph. It's a bit dirty to have everything in the 
Do a setter for config-> set mode. 
Make backward work on multivariate-should be more complicated as we'll have to register the dependencies.
E.g. x1=exp(x) + sin(y), x2 = x + x1.

## Discussion on the efficiency of backward vs forward, and especially our implementation. 
(First talk about the general efficiency of fwd vs bwd, and then our case. FWD---> We use some Variables and so on.)

### Additional: Junzhi-> Can you think about garbage management/inplace modifications ? 

In the doc, we have to precise the following:
The user can do x, y = F.unroll(X) and then define operations on those two x,y.
Can the user do x = Var(8.), y = Var(9.), and then compute operations on those two things ? 
No, We will carry unidimensional gradients and whn we do sth like adding, it will ot get that we have 
--> goes qith the discussion of the fwd mode--> seeding different input variables.

For instance: x=Var(10.), y = Var(12.). z = x + y. grad(z) will be 1+1=2 because we have single-dimensional gradients. 
Create a utils ? x,y = F.wrap([x,y]) ? Different from concat though. 


#TODO-utils so that the user can do [ x =Var([1), y=var(Var(2)), z = Var(10.)]-similar to concat ?!




