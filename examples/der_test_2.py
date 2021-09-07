# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 00:06:03 2021

@author: Sashka
"""
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import sys

sys.path.append('../')


from solver import apply_operator_set
from input_preprocessing import grid_prepare, operator_prepare

diff_results={}

def read_heat_csv(name):
    df=pd.read_csv(name,index_col=None,header=None,sep=' ')
    df=df[range(1,len(df.columns)-1)]
    rename_dict={}
    for i in range(1,len(df.columns)+1):
        rename_dict[i]='r'+str(5*i)
    df=df.rename(columns=rename_dict)  
    return df
  

mat1=read_heat_csv('Data_32_points_.dat')

r_grid=np.arange(10**(-3)*0.5,10**(-3)*0.5*(len(mat1.columns)+1),0.5*10**(-3))
t_grid=np.arange(0,0.05*len(mat1),0.05)

grad21=np.gradient(np.gradient(mat1.values,r_grid,axis=1),r_grid,axis=1)
grad20=np.gradient(np.gradient(mat1.values,t_grid,axis=0),t_grid,axis=0)
grad_df1=mat1.copy()
grad_df1[:]=grad21

grad_df0=mat1.copy()
grad_df0[:]=grad20

true_grid=[]
true_val=[]
true_grad_val0=[]
true_grad_val1=[]

for time in range(3001):
    for coord in range(1,10):
        true_grid.append([time*0.05,coord*0.5*10**(-3)])
        true_val.append(mat1['r'+str(5*coord)][time])
        true_grad_val0.append(grad_df0['r'+str(5*coord)][time])
        true_grad_val1.append(grad_df1['r'+str(5*coord)][time])
        # true_val.append((time*0.05)**2+(coord*0.5)**2)
true_grid=np.array(true_grid)
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(true_grid[:, 0], true_grid[:, 1],
                true_val, cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(true_grid[:, 0], true_grid[:, 1],
                true_grad_val1, cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(true_grid[:, 0], true_grid[:, 1],
                true_grad_val0, cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()

device = torch.device('cpu')

grid=torch.from_numpy(np.array(true_grid)).float()
data=torch.from_numpy(np.array(true_val).reshape(-1,1)).float()

# data_norm=(data-torch.min(data))/torch.max(data-torch.min(data))


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.set_title('Iteration = ' + str(t))
# ax.plot_trisurf(true_grid[:, 0], true_grid[:, 1],
#                 data_norm.numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
# ax.set_xlabel("t")
# ax.set_ylabel("r")
# plt.show()


model = torch.nn.Sequential(
    torch.nn.Linear(2, 256),
    torch.nn.Tanh(),
    # torch.nn.Dropout(0.1),
    # torch.nn.ReLU(),
    torch.nn.Linear(256, 64),
    # # torch.nn.Dropout(0.1),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 1024),
    # torch.nn.Dropout(0.1),
    torch.nn.Tanh(),
    torch.nn.Linear(1024, 1)
    # torch.nn.Tanh()
)



optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# n_epochs = 100 # or whatever
batch_size = 128 # or whatever


t=0

loss_mean=1000
min_loss=np.inf


while loss_mean>1e-5 and t<5e3:

    # X is a torch Variable
    permutation = torch.randperm(grid.size()[0])
    
    loss_list=[]
    
    for i in range(0,grid.size()[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = grid[indices], data[indices]

        # in case you wanted a semi-full example
        # outputs = model.forward(batch_x)
        # l1_lambda = 0.001
        # l1_norm =sum(p.abs().sum() for p in model.parameters())
        # loss = torch.mean(torch.abs(batch_y-model(batch_x)))+l1_lambda*l1_norm
        loss = torch.mean(torch.abs(batch_y-model(batch_x)))

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    loss_mean=np.mean(loss_list)
    if loss_mean<min_loss:
        best_model=model
        min_loss=loss_mean
    print('Surface trainig t={}, loss={}'.format(t,loss_mean))
    t+=1

# model=best_model

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(grid[:, 0].reshape(-1), grid[:, 1].reshape(-1),
                model(grid).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()


approx_val=model(grid).detach().numpy().reshape(-1)

mat1_approx=mat1.copy()


for i,point in enumerate(grid):
    time=int(point[0]/5*100)
    coord=int(point[1]/5*10*10**3)
    mat1_approx['r'+str(5*coord)][time]=approx_val[i]


        # true_val.append((time*0.05)**2+(coord*0.5)**2)
grad21_approx=np.gradient(np.gradient(mat1_approx.values,np.arange(10**(-3)*0.5,10**(-3)*0.5*(len(mat1.columns)+1),0.5*10**(-3)),axis=1),np.arange(10**(-3)*0.5,10**(-3)*0.5*(len(mat1.columns)+1),0.5*10**(-3)),axis=1)
grad20_approx=np.gradient(np.gradient(mat1_approx.values,np.arange(0,0.05*len(mat1),0.05),axis=0),np.arange(0,0.05*len(mat1),0.05),axis=0)

grad_df1_approx=mat1.copy()
grad_df1_approx[:]=grad21_approx

grad_df0_approx=mat1.copy()
grad_df0_approx[:]=grad20_approx

true_grid=[]
true_val_approx=[]
true_grad_val0_approx=[]
true_grad_val1_approx=[]

for time in range(3001):
    for coord in range(1,10):
        true_grid.append([time*0.05,coord*0.5*10**(-3)])
        true_val_approx.append(mat1_approx['r'+str(5*coord)][time])
        true_grad_val0_approx.append(grad_df0_approx['r'+str(5*coord)][time])
        true_grad_val1_approx.append(grad_df1_approx['r'+str(5*coord)][time])
        # true_val.append((time*0.05)**2+(coord*0.5)**2)
true_grid=np.array(true_grid)
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(true_grid[:, 0], true_grid[:, 1],
                true_val_approx, cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(true_grid[:, 0], true_grid[:, 1],
                true_grad_val1_approx, cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(true_grid[:, 0], true_grid[:, 1],
                true_grad_val0_approx, cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()





true_grid=np.array(true_grid)


prepared_grid = grid_prepare(grid)

operator = {
    '1*d2u/dx2**1':
        {
            'coeff': 1,
            'd2u/dx2': [0,0],
            'pow': 1
        }
}

operator = operator_prepare(operator, prepared_grid, subset=None, true_grid=grid, h=10**(-1))


op_clean = apply_operator_set(model, operator)

grad20_array=np.array(true_grad_val0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(true_grid[:, 0], true_grid[:, 1],
                grad20_array, cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(prepared_grid[:, 0].reshape(-1), prepared_grid[:, 1].reshape(-1),
                op_clean.detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()


diff_results['clean']=op_clean
