import numpy as np
import matplotlib.pyplot as plt
from holoviews.operation import gradient

n_iter=100
x_init=0
alpha=1e-2
x_values=[x_init]
x=float(x_init)
gradients=[]
for n in range(n_iter):
    f_grad=2*(x-1)**5-5*(x-2)**4+20*x**3
    x=x-alpha*f_grad
    x_values.append(x)
    gradients.append(f_grad)
    x=x_values[-1]
    print(x)