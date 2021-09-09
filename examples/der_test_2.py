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
    for i in range(0,len(df.columns)):
        rename_dict[i+1]='r'+str(i)
    df=df.rename(columns=rename_dict)  
    return df
  




def heat_second_FD(mat,x_grid,y_grid):
    grad20=np.gradient(np.gradient(mat1.values,x_grid,axis=0),t_grid,axis=0)
    grad21=np.gradient(np.gradient(mat1.values,y_grid,axis=1),r_grid,axis=1)
    
    grad_df0=mat1.copy()
    grad_df0[:]=grad20
    
    grad_df1=mat1.copy()
    grad_df1[:]=grad21
    

    return grad_df0,grad_df1




def flatten_df_to_list(df,nx,ny,grid_x,grid_y):
    true_grid=[]
    true_val=[]
    for time in range(nx):
        for coord in range(ny):
              true_grid.append([grid_x[time],grid_y[coord]])      
              true_val.append(df['r'+str(coord)][time])
    true_grid=np.array(true_grid)
    return true_grid,true_val



def plot_list(true_grid,true_val,title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_title('Iteration = ' + str(t))
    ax.plot_trisurf(true_grid[:, 0], true_grid[:, 1],
                    true_val, cmap=cm.jet, linewidth=0.2, alpha=1)
    ax.set_xlabel("t")
    ax.set_ylabel("r")
    ax.set_title(title)
    plt.show()


mat1=read_heat_csv('Data_32_points_.dat')

# r_grid=np.arange(10**(-3)*0.5,10**(-3)*0.5*(len(mat1.columns)+1),0.5*10**(-3))

# r_grid=np.arange(0.5,0.5*(len(mat1.columns)+1),0.5)

r_grid=np.linspace(0,1,len(mat1.columns))
# t_grid=np.arange(0,0.05*len(mat1),0.05)
t_grid=np.linspace(0,1,len(mat1))
true_grid,true_val=flatten_df_to_list(mat1,3000,9,t_grid,r_grid)
        

grad_df0,grad_df1=heat_second_FD(mat1,t_grid,r_grid)

_,true_grad_val0=flatten_df_to_list(grad_df0,3000,9,t_grid,r_grid)        
_,true_grad_val1=flatten_df_to_list(grad_df1,3000,9,t_grid,r_grid)

plot_list(true_grid,true_val,title='Initial field')

plot_list(true_grid,true_grad_val0,title='FD field d2T/dt2')

plot_list(true_grid,true_grad_val1,title='FD field d2T/dr2')



device = torch.device('cpu')

grid=torch.from_numpy(true_grid).float()
data=torch.from_numpy(np.array(true_val).reshape(-1,1)).float()



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


def train_minibatch(model,optimizer,grid,data,batch_size=128,tmax=5e3):
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
    return best_model

model=train_minibatch(model,optimizer,grid,data,batch_size=128,tmax=5e3)

approx_val=model(grid).detach().numpy().reshape(-1)

plot_list(true_grid,approx_val,title='Approximate field')


def approx_val_to_df(field_df,approx_val,grid):

    df_approx=field_df.copy()
    
    dt=np.unique(grid[:,0])[1]-np.unique(grid[:,0])[0]
    dr=np.unique(grid[:,1])[1]-np.unique(grid[:,1])[0]
    
    for i,point in enumerate(grid):
        time=int(point[0]/dt)
        coord=int(point[1]/dr)-int(torch.min(grid[:,1])/dr)
        df_approx['r'+str(coord)][time]=approx_val[i]
    return df_approx

mat1_approx=approx_val_to_df(mat1,approx_val,grid)

grad_df0_approx,grad_df1_approx=heat_second_FD(mat1_approx,t_grid,r_grid)

_,true_val_approx=flatten_df_to_list(mat1_approx,3000,9,t_grid,r_grid)

_,true_grad_val0_approx=flatten_df_to_list(grad_df0_approx,3000,9,t_grid,r_grid)        
_,true_grad_val1_approx=flatten_df_to_list(grad_df1_approx,3000,9,t_grid,r_grid)


plot_list(true_grid,true_grad_val0_approx,title='FD approx field d2T/dt2')

plot_list(true_grid,true_grad_val1_approx,title='FD approx field d2T/dr2')



prepared_grid = grid_prepare(grid)
for h_step in [0.0001,0.0005,0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.5,1]:
    operator = {
        '1*d2u/dx2**1':
            {
                'coeff': 1,
                'd2u/dx2': [0,0],
                'pow': 1
            }
    }
    
    operator = operator_prepare(operator, prepared_grid, subset=None, true_grid=grid, h=h_step)
    
    
    op_clean = apply_operator_set(model, operator)
    
    nn_op_field=op_clean.detach().numpy().reshape(-1)
    
    plot_list(prepared_grid,nn_op_field,title='NN derivative field d2T/dt2 h={}'.format(h_step))
    
    diff_results['clean']=op_clean
    
    # error=torch.mean((torch.tensor(true_grad_val0).reshape(-1,1)-op_clean)**2)
    # print('d2T/dt2 h={}, error={}'.format(h_step,error))


for h_step in [5e-4,6e-4,7e-4,8e-4,9e-4,1e-3,5e-3,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.5,1]:
    operator = {
        '1*d2u/dx2**1':
            {
                'coeff': 1,
                'd2u/dx2': [1,1],
                'pow': 1
            }
    }
    
    operator = operator_prepare(operator, prepared_grid, subset=None, true_grid=grid, h=h_step)
    
    
    op_clean = apply_operator_set(model, operator)
    
    nn_op_field=op_clean.detach().numpy().reshape(-1)
    
    plot_list(prepared_grid,nn_op_field,title='NN derivative field d2T/dr2 h={}'.format(h_step))
    
    diff_results['clean']=op_clean
    
    # error=torch.mean((torch.tensor(true_grad_val1).reshape(-1,1)-op_clean)**2)
    # print('d2T/dr2 h={}, error={}'.format(h_step,error))

operator = {
    '1*d2u/dx2**1':
        {
            'coeff': 1,
            'd2u/dx2': [0,0],
            'pow': 1
        }
}

operator = operator_prepare(operator, prepared_grid, subset=None, true_grid=grid, h=8.333333e-05)


op_clean = apply_operator_set(model, operator)

nn_op_field=op_clean.detach().numpy().reshape(-1)

plot_list(prepared_grid,nn_op_field,title='NN derivative field d2T/dt2 h={}'.format(8.333333e-05))

diff_results['clean']=op_clean

# error=torch.mean((torch.tensor(true_grad_val0).reshape(-1,1)-op_clean)**2)
# print('d2T/dt2 h={}, error={}'.format(h_step,error))