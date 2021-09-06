# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:55:46 2021

@author: user
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

mat1=pd.read_csv('Data_32_points_.dat',index_col=None,header=None,sep=' ')
mat1=mat1[range(1,33)]

rename_dict={}
for i in range(1,33):
    rename_dict[i]='r'+str(5*i)

mat1=mat1.rename(columns=rename_dict)

rename_dict={}
for i in range(1,len(mat1.columns)):
    rename_dict[i]='r'+str(5*i)

mat1=mat1.rename(columns=rename_dict)

true_grid=[]
true_val=[]

for time in range(3001):
    for coord in range(1,10):
        true_grid.append([time*0.05,coord*0.5])
        true_val.append(mat1['r'+str(5*coord)][time])
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


device = torch.device('cpu')

grid=torch.from_numpy(np.array(true_grid)).float()
data=torch.from_numpy(np.array(true_val).reshape(-1,1)).float()

# data_norm=(data-torch.min(data))/torch.max(data-torch.min(data))

data_norm=data

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.set_title('Iteration = ' + str(t))
# ax.plot_trisurf(true_grid[:, 0], true_grid[:, 1],
#                 data_norm.numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
# ax.set_xlabel("t")
# ax.set_ylabel("r")
# plt.show()



grid.to(device)
data_norm.to(device)

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

# l1_lambda = 0.001
# l1_norm =sum(p.abs().sum() for p in model.parameters())



# n_epochs = 100 # or whatever
batch_size = 128 # or whatever


t=0

loss_mean=1000

while loss_mean>1e-5 and t<1e3:

    # X is a torch Variable
    permutation = torch.randperm(grid.size()[0])
    
    loss_list=[]
    
    for i in range(0,grid.size()[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = grid[indices], data_norm[indices]

        # in case you wanted a semi-full example
        # outputs = model.forward(batch_x)
        loss = torch.mean(torch.abs(batch_y-model(batch_x)))

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    loss_mean=np.mean(loss_list)
    print('Surface trainig t={}, loss={}'.format(t,loss_mean))
    t+=1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(grid[:, 0].reshape(-1), grid[:, 1].reshape(-1),
                model(grid).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()


prepared_grid = grid_prepare(grid)

operator = {
    '1*d2u/dx2**1':
        {
            'coeff': 1,
            'd2u/dx2': [1,1],
            'pow': 1
        }
}

operator = operator_prepare(operator, prepared_grid, subset=None, true_grid=grid, h=0.1)


op_clean = apply_operator_set(model, operator)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(prepared_grid[:, 0].reshape(-1), prepared_grid[:, 1].reshape(-1),
                op_clean.detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()


diff_results['clean']=op_clean




for smth in ['0','01','03','05','07','1','3','5','7','10']:

    mat1=pd.read_csv('Noise_1/Data_20_points_'+smth+'.dat',index_col=None,header=None,sep=' ',skiprows=3)
    mat1=mat1[range(1,len(mat1.columns))]
    
    rename_dict={}
    for i in range(1,len(mat1.columns)):
        rename_dict[i]='r'+str(5*i)
    
    mat1=mat1.rename(columns=rename_dict)
    
    true_grid=[]
    true_val=[]
    
    for time in range(3001):
        for coord in range(1,10):
            true_grid.append([time*0.05,coord*0.5])
            true_val.append(mat1['r'+str(5*coord)][time])
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
    
    
    device = torch.device('cpu')
    
    grid=torch.from_numpy(np.array(true_grid)).float()
    data=torch.from_numpy(np.array(true_val).reshape(-1,1)).float()
    
    # data_norm=(data-torch.min(data))/torch.max(data-torch.min(data))
    
    data_norm=data
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.set_title('Iteration = ' + str(t))
    # ax.plot_trisurf(true_grid[:, 0], true_grid[:, 1],
    #                 data_norm.numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
    # ax.set_xlabel("t")
    # ax.set_ylabel("r")
    # plt.show()
    
    
    
    grid.to(device)
    data_norm.to(device)
    
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(2, 256),
    #     torch.nn.Tanh(),
    #     # torch.nn.Dropout(0.1),
    #     # torch.nn.ReLU(),
    #     torch.nn.Linear(256, 64),
    #     # # torch.nn.Dropout(0.1),
    #     torch.nn.Tanh(),
    #     torch.nn.Linear(64, 1024),
    #     # torch.nn.Dropout(0.1),
    #     torch.nn.Tanh(),
    #     torch.nn.Linear(1024, 1)
    #     # torch.nn.Tanh()
    # )
    
    
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # l1_lambda = 0.001
    # l1_norm =sum(p.abs().sum() for p in model.parameters())
    
    
    
    # n_epochs = 100 # or whatever
    batch_size = 128 # or whatever
    
    
    t=0
    
    loss_mean=1000
    
    while loss_mean>1e-5 and t<1e3:
    
        # X is a torch Variable
        permutation = torch.randperm(grid.size()[0])
        
        loss_list=[]
        
        for i in range(0,grid.size()[0], batch_size):
            optimizer.zero_grad()
    
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = grid[indices], data_norm[indices]
    
            # in case you wanted a semi-full example
            # outputs = model.forward(batch_x)
            loss = torch.mean(torch.abs(batch_y-model(batch_x)))
    
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_mean=np.mean(loss_list)
        print('Surface trainig t={}, loss={}'.format(t,loss_mean))
        t+=1
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Iteration = ' + str(t))
    ax.plot_trisurf(grid[:, 0].reshape(-1), grid[:, 1].reshape(-1),
                    model(grid).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
    ax.set_xlabel("t")
    ax.set_ylabel("r")
    plt.show()
    
    
    prepared_grid = grid_prepare(grid)
    
    operator = {
        '1*d2u/dx2**1':
            {
                'coeff': 1,
                'd2u/dx2': [1,1],
                'pow': 1
            }
    }
    
    operator = operator_prepare(operator, prepared_grid, subset=None, true_grid=grid, h=0.1)
    
    
    diff_field = apply_operator_set(model, operator)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Iteration = ' + str(t))
    ax.plot_trisurf(prepared_grid[:, 0].reshape(-1), prepared_grid[:, 1].reshape(-1),
                    diff_field.detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
    ax.set_xlabel("t")
    ax.set_ylabel("r")
    plt.show()
    
    
    diff_results[smth]=diff_field


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(prepared_grid[:, 0].reshape(-1), prepared_grid[:, 1].reshape(-1),
                (diff_results['clean']).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()

for smth in ['0','01','03','05','07','1','3','5','7','10']:
    err=torch.mean((diff_results['clean']-diff_results[smth])**2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Iteration = ' + str(t))
    ax.plot_trisurf(prepared_grid[:, 0].reshape(-1), prepared_grid[:, 1].reshape(-1),
                    (diff_results[smth]).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
    ax.set_xlabel("t")
    ax.set_ylabel("r")
    plt.show()
    print('Smth={}, Error = {}'.format(smth,err))
