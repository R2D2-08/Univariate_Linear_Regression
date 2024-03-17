def regression(new_x,new_m,new_b):
    return new_m*new_x+new_b
def mse(new_y_dash,new_y):
    sigma=torch.sum((new_y_dash-new_y)**2)
    return (1/len(new_y))*sigma
def mse_dash_wrt_m(new_y_dash,new_y,new_x):
    sig=2*torch.sum((new_y_dash-new_y)*new_x)
    return (1/len(new_y))*sig
def mse_dash_wrt_b(new_y_dash,new_y):
    sig=2*torch.sum((new_y_dash-new_y))
    return (1/len(new_y))*sig
def update_values(m,b,lr,new_y_dash,new_y,new_x):
    m=m-lr*mse_dash_wrt_m(new_y_dash,new_y,new_x)
    b=b-lr*mse_dash_wrt_b(new_y_dash,new_y)
    return m,b
import torch 
import numpy as np
import matplotlib.pyplot as plt
x=torch.tensor(np.arange(0,20))
y=torch.tensor([10.5,8.9,8.5,7.05,6.5,5,4.1,2.5,2.,1.15,0,-1.5,-2.5,-3.,-4.5,-5,-6.5,-6.9,-8.3,-9.5])
m=torch.tensor([5.]).requires_grad_()
b=torch.tensor([0.1]).requires_grad_()
epochs=1000
for epoch in range(epochs):
    y_dash=regression(x,m,b)
    C=mse(y_dash,y)
    m,b=update_values(m,b,0.007,y_dash,y,x)
    print(f"epoch : {epoch}, m = {m.item()}, b = {b.item()}, C = {C.item()}")    
plt.plot([0,20],[b.item(),(m.item()*20+b.item())])
plt.scatter(x,y)
plt.show()