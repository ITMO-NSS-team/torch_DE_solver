# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 18:07:37 2021

@author: user
"""
import numpy as np
import torch


device = torch.device('cpu')

true_grid = np.linspace(1/4, 7/4, 502)

p_4_dat=np.genfromtxt("wolfram_sln/p_IV_sln_501.csv",delimiter=',')

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

model = torch.nn.Sequential(
    torch.nn.Linear(1, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 1)
    )

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# l1_lambda = 0.001
# l1_norm =sum(p.abs().sum() for p in model.parameters())



# n_epochs = 100 # or whatever
batch_size = 128 # or whatever


t=0

loss_mean=1000
min_loss=np.inf


while loss_mean>1e-5 and t<1e5:

    # X is a torch Variable
    permutation = torch.randperm(grid.size()[0])
    
    loss_list=[]
    
    for i in range(0,grid.size()[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = grid[indices], data[indices]

        # in case you wanted a semi-full example
        # outputs = model.forward(batch_x)
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

model=best_model


torch.save({'model':model, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, '../cache/p_IV_test.tar')
