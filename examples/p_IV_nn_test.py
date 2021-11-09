# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 18:07:37 2021

@author: user
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

# sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))


import numpy as np
import torch
import torch_rbf as rbf
import matplotlib.pyplot as plt

device = torch.device('cpu')

true_grid = np.linspace(1/4, 7/4, 502)

p_4_dat_init=np.genfromtxt("wolfram_sln/p_IV_sln_501.csv",delimiter=',')

# p_4_dat=p_4_dat_init/np.max(p_4_dat_init)

p_4_dat=p_4_dat_init


grid=torch.from_numpy(np.array(true_grid).reshape(-1,1)).float()
data=torch.from_numpy(np.array(p_4_dat).reshape(-1,1)).float()

grid.to(device)
data.to(device)

# model = torch.nn.Sequential(
#     torch.nn.Linear(1, 256),
#     torch.nn.Tanh(),
#     # torch.nn.Dropout(0.1),
#     # torch.nn.ReLU(),
#     torch.nn.Linear(256, 64),
#     # # torch.nn.Dropout(0.1),
#     torch.nn.Tanh(),
#     torch.nn.Linear(64, 1024),
#     # torch.nn.Dropout(0.1),
#     torch.nn.Tanh(),
#     torch.nn.Linear(1024, 64),
#     torch.nn.Tanh(),
#     torch.nn.Linear(64, 256),
#     torch.nn.Tanh(),
#     torch.nn.Linear(256, 1)
#     # torch.nn.Tanh()
# )





# l1_lambda = 0.001
# l1_norm =sum(p.abs().sum() for p in model.parameters())



# n_epochs = 100 # or whatever
batch_size = 128 # or whatever

exp_list=[]

for _ in range(10):
    
    model = torch.nn.Sequential(
    torch.nn.Linear(1, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 1)
    # torch.nn.Tanh()
    )
    
    params = list(model.parameters())


    optimizer = torch.optim.Adam(params, lr=0.0001)
    
    
    t=0
    
    loss_mean=1000
    min_loss=np.inf
    
    
    t_init=t
    t_end=t
    
    while min_loss>1e-5 and t<1e5:
    
        # X is a torch Variable
        permutation = torch.randperm(grid.size()[0])
        
        loss_list=[]
        
        for i in range(0,grid.size()[0], batch_size):
            optimizer.zero_grad()
    
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = grid[indices], data[indices]
    
            # in case you wanted a semi-full example
            # outputs = model.forward(batch_x)
            loss = torch.mean(batch_x*(batch_y-model(batch_x))**2)
            # loss = torch.mean((batch_y-model(batch_x))**2)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_mean=np.mean(loss_list)
        if loss_mean<min_loss:
            best_model=model
            min_loss=loss_mean
            t_end=t    
            l2_norm=torch.mean((best_model(grid)-data)**2)
            # print('t={} steps_taken={}, min_loss={}, l2_norm={}'.format(t, t_end-t_init,min_loss,l2_norm))
        t_init=t_end
        t+=1
    
    # model=best_model
    
    l2_norm=torch.mean((best_model(grid)-data)**2)
    
    print('Best model for t_steps={}  min_loss={}, l2_norm={}'.format(t ,min_loss,l2_norm))
    exp_list.append({'type': 'weighted_l2','l2_norm':l2_norm.item()})


for _ in range(10):
    
    model = torch.nn.Sequential(
    torch.nn.Linear(1, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 1)
    # torch.nn.Tanh()
    )
    
    params = list(model.parameters())


    optimizer = torch.optim.Adam(params, lr=0.0001)
    
    t=0
    
    loss_mean=1000
    min_loss=np.inf
    
    
    t_init=t
    t_end=t
    
    while min_loss>1e-5 and t<1e5:
    
        # X is a torch Variable
        permutation = torch.randperm(grid.size()[0])
        
        loss_list=[]
        
        for i in range(0,grid.size()[0], batch_size):
            optimizer.zero_grad()
    
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = grid[indices], data[indices]
    
            # in case you wanted a semi-full example
            # outputs = model.forward(batch_x)
            # loss = torch.mean(batch_x*(batch_y-model(batch_x))**2)
            loss = torch.mean((batch_y-model(batch_x))**2)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_mean=np.mean(loss_list)
        if loss_mean<min_loss:
            best_model=model
            min_loss=loss_mean
            t_end=t    
            l2_norm=torch.mean((best_model(grid)-data)**2)
            # print('t={} steps_taken={}, min_loss={}, l2_norm={}'.format(t, t_end-t_init,min_loss,l2_norm))
        t_init=t_end
        t+=1
    
    # model=best_model
    
    l2_norm=torch.mean((best_model(grid)-data)**2)
    
    print('Best model for t_steps={}  min_loss={}, l2_norm={}'.format(t ,min_loss,l2_norm))
    exp_list.append({'type': 'l2','l2_norm':l2_norm.item()})

exp_list_new=[]

for exp in exp_list:
    exp_list_new.append({'type':exp['type'],'l2_norm':exp['l2_norm'].item()})
# fig = plt.figure()
# plt.scatter(grid.reshape(-1), best_model(grid).detach().numpy().reshape(-1))
# # analytical sln is 1/2*(-1 + 3*t**2)
# plt.scatter(grid.reshape(-1), p_4_dat.reshape(-1))
# plt.show()


# torch.save({'model':model, 'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict()}, '../cache/p_IV_test.tar')
