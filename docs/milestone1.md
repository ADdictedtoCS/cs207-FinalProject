
# Milestone 1
## CS207 Final Project, Group 28
#### Team Members:
* Josh Bodner  
* Théo Guenais  
* Daiki Ina  
* Junzhi Gong

# Introduction

Ever since the notion of a derivative was first defined in the days of Newton and Leibniz, differentiation has become a crucial component across quantitative analyses. Today, differentiation is applied within a wide range of scientific disciplines from cell biology to electrical engineering and astrophysics. Differentiation is also central to the domain of optimization where it is applied to find maximum and minimum values of functions.

Because of the widespread use and applicability of differentiation, it would be highly beneficial if scientists could efficiently calculate derivatives of functions using a Python package. Such a package could save scientists time and energy from having to compute derivatives symbolically, which can be especially complicated if the function of interest has vector inputs and outputs.

Our Python package, autodiff, will address this need by allowing the user to implement the forward mode (and potentially the reverse mode) of automatic differentiation (AD). Using AD, we are able to calculate derivatives to machine precision in a manner that is less costly than symbolic differentiation.

# Background
In this section we provide a brief overview of the mathematical concepts relevant to our implementation:

##### 1. Multivariable Differental Calculus:

Let $f: \mathbb{R^m} \rightarrow \mathbb{R^n}: x \mapsto f(x)$. Under certain regularity and smoothness assumptions, we define the derivative of f as $f': \mathbb{R^m} \rightarrow L_{\mathbb{R^m}, \mathbb{R^n}}$, with $L_{\mathbb{R^m}, \mathbb{R^n}}$ being the space of the linear mapping $\mathbb{R^m}\rightarrow \mathbb{R^n}$ which can be transformed to $\mathbb{R^{mxn}}$.

*This general definition might be broken due to lack of smoothness, in which case, we refer to "directional derivatives" and specifically "Gateaux-derivatives".*

From this lense, we can understand the derivative of a function evaluated at a given point to be a matrix. When this function is real-valued, we can obtain a vector $\nabla_x f \in \mathbb{R^m}$.

Specifically, the gradient of a scalar-valued multivariable function is a vector of its partial derivatives with respect to each input:

$\nabla f = 
  \begin{bmatrix}
    \frac{\partial f}{\partial x_1} \\
    \frac{\partial f}{\partial x_2} \\
    \frac{\partial f}{\partial x_3} \\
    \vdots
  \end{bmatrix}$

##### 2. Chain Rule
Let $f: \mathbb{R^m} \rightarrow \mathbb{R^n}$ and $g: \mathbb{R^p} \rightarrow \mathbb{R^m}$. 
Then $f \circ g: \mathbb{R^p} \rightarrow \mathbb{R^n}$ is such that (under regularity assumptions), $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$. This operational rule, is known as the chain rule.

---

*Example:*  
$ y = sin(e^{x})$  
$ g(x) = e^{x}$  
$ f(x) = sin(g(x))$  
 
$(f \circ g)'(x) = f'(g(x))g'(x) = cos(e^{x})e^{x}$

##### 3. Automatic Differentiation, Graph Structure, and Elementary Functions

Automatic differentiation is a process for evaluating derivatives computationally at a particular evaluation point. Specifically, the forward mode of automatic differentiation calculates the product of the gradient with a seed vector $ \nabla f \cdot p $ or the product of the Jacobian with a seed vector if f is a vector $Jp$. Automatic differentiation can compute derivatives to machine precision.

In the forward mode of automatic differentiation, the calculation is done in steps where the partial derivatives and values are computed at each step of the computational graph.

The edges of the computational graph are such that transitioning from node to node involves simple operations or calculation of elementary functions such as $sin(x)$, $cos(x)$, $e^{x}$, etc.

---
*Example:*  
$ y = sin(e^{x}) + x$  
Calculate $\frac{dy}{dx}$ at x=5 

The evaluation trace of the forward mode of AD is as follows: 

| Trace | Elementary Function | Current Value |  Derivative          | Derivative Value |
|-------|---------------------|---------------|----------------------|------------------|
| $x_1$ |        $x_1$        |    5          | $\dot{x_1}$          |        1         |
| $x_2$ |        $e^{x_1}$    |    148.413    | $e^{x_1}\dot{x_1}$   |        148.413   |
| $x_3$ |        $sin(x_2)$   |    0.524      | $cos(x_2)\dot{x_2}$  |       -126.425   |
| $x_4$ |        $x_3 + x_1$  |    5.524      | $\dot{x_3}+\dot{x_1}$|       -125.425   |


The computational graph can be seen below:  
![alt text](figs\milestone1_graph.png "Title")

# How to use autodiff

The user will be able to import our package as follows:


```python
import autodiff as AD
```

Next, we are considering 2 possibilities for how the user will interact with our package.

We will be working out which option makes more sense as we begin the process of development, however, option 1 seems like the natural first step whereas the second option would likely be a follow-up implementation.


```python
# --- Option 1 ---

# The user could instantiate a variable at a particular evaluation point.
# Next, a function could be expressed as a composition of the elementary functions..

x = AD.variable(5) # Evaluate at point x = 5
y = AD.function.exp(x)
y.gradient
>>> 148.413159
```


```python
# --- Option 2 ---

# The user could instantiate autodiff variables and create functions by
# performing operations on those autodiff variables.
# They could then call get_der to get the derivative.

x = AD.variable(length of input vector)
y = AD.variable(length of input vector)

my_function = 4 * x + sqrt(x) + x * y

function.get_derivative(target=x, x=2, y=3)
>>> 7.3535533
```

# Software Organization

##### Directory Structure:

We will have our main implementation stored in the autodiff directory. This is where modules for our implementation of the forward mode of AD will be located. The autodiff directory will also have a subdirectory containing tests.

```
cs207-FinalProject/
    docs/
        milestone1.ipynb
        ...
    autodiff/
        __init__.py
        function.py
        variable.py
        utils.py
        tests/
            __init__.py
            test_function.py
            test_variable.py
            test_utils.py
            ...
    README.md
    requirements.txt
    
```
##### Modules to Include:
We will include a function module containing our function class as well as the implementation of our elementary functions that constitute the computational graph. We may end up breaking these out into separate modules for better readability. The aforementioned structure is not exhaustive, and we may expand on it as we begin writing our code.

For example, we may also end up implementing the reverse mode of AD, in which case this would potentially be stored as a separate module within the autodiff directory.

##### Testing:

Our test suite will live within the autodiff directory (see above). Additionally for continuous integration and code coverage we will be utilizing both TravisCI and CodeCov. We have already set up basic functionality for TravisCI and CodeCov for our repository. Related documentation will also be available to the user.

##### Distribution and Packaging:

We will distribute our package using PyPI (upload with Twine).
The user can then install our package via the command line: "pip install autodiff"
 
##### Other Considerations:

We may choose to build a GUI in order to make our package more accesible to end users with limited Python coding abilities. The possibility of building a static graph in order to repeat and speed up the evaluation of a given derivative at several points is also an option we are considering. 

# Implementation

##### Core Data Structures:

For each function defined by users, we will create a Directed Acyclic Graph (DAG) data structure, which contains all computing nodes for the forward mode of AD (and potentially the reverse mode of AD).

Our operation will be based upon two main concepts: 
* **AD.variable** carries the information flow through the computational graph. (see below).
* **AD.function** implements the elementary functions and constitutes the edges (or nodes, depending on ones's interpretation of the computational graph).

Within each node, there will be a list of values and derivatives, which could be used to complete the forward mode (and potentially reverse mode) AD computation. Therefore, the lists from all nodes form the computation table in either forward mode AD or reverse mode AD. We may also include some other metadata along with those variables.

##### Classes:

The classes we will implement include the **Function** (possibly both a forward and reverse mode version), and **Variable**. Additionally, we may implement the elementary functions as subclasses of **Function**.

##### Methods and name attributes:

Users could start by creating **AD.variable**s, which could either take scalar or vector values. For example, users could call **x = AD.variable(init)** to create a variable, for any multidimensional vector **init**.

Users could then create a function by applying operations on **AD.variable**s. The basic operations includes "+" "-" "\*" "/", as well as the elementary functions, including **AD.function.exp()**, **AD.function.sin()**, **AD.function.log()**, *e.t.c.* For example, a user could create a function in the following way:

```Python
def my_func(x):
    x1 = AD.function.sin(x)
    x2 = AD.function.exp(x1)
    return AD.function.cos(x1) + x2
 
x = AD.variable(1000)
y = my_func(x)
y.grad
```

The **function** class will have the methods **get_val()** and **get_der()** to get its value and derivative.

**AD.function**  could also include the following attributes, but none of them is necessary for the first version of the code and the rudimentary implementation.

- val (Current value)-in that case, we would have to make sure that a function points to a Variable.
- der (Current derivative)
- children / graph (Record the graph structure)
- val_list (Value table for forward mode and reverse mode)
- der_list (Derivative table for forward mode and reverse mode)
- mode (Forward or reverse)


#### External dependencies:
Required:
- Numpy (for vector operations)

Additional possibilities:  
- Math (for scientific math functions)
- Matplotlib (for graphing if we end up building a GUI)

#### Elementary functions:

We will likely implement the elementary functions such as **sin**, **sqrt**, **log**, **exp**, etc. as *subclasses* of **function**. 

Elementary function classes will likely have similar features to **AD.function**s, but we will need to determine their internal functionalities, including how to calculate their derivatives.

#### Additional Implementation Details:

How we will use the two components "Variable" and "Function", to achieve information flow:

1. Concept of “Variable” 
    - The variables carry the information flow, which will be used when performing computation on the graph in the forward mode or the reverse mode of AD. The variable would have ```val``` (value) and ```grad``` (gradient) attributes, consistent with the trace of the forward mode operations.  
       
       
2. Concept of "Function"
    - The Function implements the atomic operations of the graph. It would take as input a ```AD.variable``` and outputs an object of the same type, with the updated ```val``` and ```grad``` attributes. 
    
Below is pseudocode to show how a function brings information flow using a variable. The code is rudimentary and does not take into account the parent class, static methods,... This demonstration is for illustrative purposes only.


```python
import autodiff as AD

x = AD.Variable(0)
y = AD.function.exp(x)

y.grad
>>> 1

y.val
>>> 1 

#Internally
class exp():
    def get_val(self, x):
        return numpy.exp(x)
    
    def get_der(self, x):
        return numpy.exp(x)
    
    def __call__(self, x:AD.variable)->AD.variable
        output = AD.variable(self.get_val(x.val) #Gradient initialized at np.ones(len(input))
        output.grad = self.get_derivative(x.val) * x.grad #Chain rule
        return output 
                             

# To simplify things, we could also have the chain rule implemented for the entire class (static method).
```
