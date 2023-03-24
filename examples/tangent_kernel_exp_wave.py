import torch
import numpy as np
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def nn_autograd_simple(model, points, order,axis=0):
    points.requires_grad=True
    f = model(points).sum()
    for i in range(order):
        grads, = torch.autograd.grad(f, points, create_graph=True)
        f = grads[:,axis].sum()
    return grads[:,axis]


def nn_autograd_mixed(model, points,axis=[0]):
    points.requires_grad=True
    f = model(points).sum()
    for ax in axis:
        grads, = torch.autograd.grad(f, points, create_graph=True)
        f = grads[:,ax].sum()
    return grads[:,axis[-1]]



def nn_autograd(*args,axis=0):
    model=args[0]
    points=args[1]
    if len(args)==3:
        order=args[2]
        grads=nn_autograd_simple(model, points, order,axis=axis)
    else:
        grads=nn_autograd_mixed(model, points,axis=axis)
    return grads.reshape(-1,1)



x = torch.from_numpy(np.linspace(0, 1, 11))
t = torch.from_numpy(np.linspace(0, 1, 11))

coord_list = []
coord_list.append(x)
coord_list.append(t)

grid = torch.cartesian_prod(x, t).float()


# Initial conditions at t=0
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
    
# u(0,x)=sin(pi*x)
bndval1 = torch.sin(np.pi * bnd1[:, 0])
    
# Initial conditions at t=1
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([1], dtype=np.float64))).float()
    
# u(1,x)=sin(pi*x)
bndval2 = torch.sin(np.pi * bnd2[:, 0])
    
# Boundary conditions at x=0
bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()
    
# u(0,t)=0
bndval3 = torch.from_numpy(np.zeros(len(bnd3), dtype=np.float64))
    
# Boundary conditions at x=1
bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()
    
# u(1,t)=0
bndval4 = torch.from_numpy(np.zeros(len(bnd4), dtype=np.float64))

model = torch.nn.Sequential(
    torch.nn.Linear(2, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


def wave_op(model,points):
      return torch.mean((4*nn_autograd_simple(model, points, 2,axis=0)-nn_autograd_simple(model, points, 2,axis=1))**2)


def bnd_op(model):
    bnd=torch.cat((model(bnd1),model(bnd2),model(bnd3),model(bnd4)))
    bndval=torch.cat((bndval1,bndval2,bndval3,bndval4)).reshape(-1,1)
    return torch.mean(torch.abs(bnd-bndval))


#print(grid.shape)



#K_uu,K_rr,K=compute_K(model)



def compute_K(model):

    optimizer.zero_grad()

    K_uu=torch.zeros(grid.shape[0])

    for i in range(grid.shape[0]):
            #print("i={},j={}".format(i,j))
            w_loss_1=wave_op(model,grid[i].reshape(1,-1))
            w_loss_1.backward()
            weights_1 = [w.grad.reshape(-1) if w.grad is not None else torch.tensor([0]) for w in model.parameters()]
            weights_1_cat=torch.cat(weights_1)

            K_uu[i]=torch.dot(weights_1_cat,weights_1_cat)
            optimizer.zero_grad()



    bnd_pts=torch.cat((bnd1,bnd2,bnd3,bnd4))

    K_rr=torch.zeros(bnd_pts.shape[0])

    for i in range(bnd_pts.shape[0]):
            #print("i={},j={}".format(i,j))
            w_loss_1=model(bnd_pts[i])
            w_loss_1.backward()
            weights_1 = [w.grad.reshape(-1) if w.grad is not None else torch.tensor([0]) for w in model.parameters()]
            weights_1_cat=torch.cat(weights_1)

            K_rr[i]=torch.dot(weights_1_cat,weights_1_cat)
            optimizer.zero_grad()
    
    return K_uu,K_rr




def compute_adaptive_lambdas(model):
    K_uu,K_rr=compute_K(model)
    slb=K_uu.sum()
    slr=K_rr.sum()

    lb=(slb+slr)/slb

    lr=(slb+slr)/slr

    return lb,lr

#lb,lr=compute_adaptive_lambdas(model)

#t=0

#print('t={}, lb={},lr={}'.format(t,lb,lr))

#lb,lr=compute_adaptive_lambdas_2(model)

#t=0

#print('t={}, lb={},lr={}'.format(t,lb,lr))

lb,lr=compute_adaptive_lambdas(model)

t=0

print('t={}, lb={},lr={}'.format(t,lb,lr))


from copy import deepcopy

grid1=deepcopy(grid)

def closure():
    #nonlocal cur_loss
    optimizer.zero_grad()
    loss =lr*wave_op(model, grid1)+lb*bnd_op(model)
    loss.backward()
    #cur_loss = loss.item()
    return loss

curr_loss=10e5

while abs(lb-lr)>1e-3:
    loss=optimizer.step(closure)
    curr_loss=loss.item()
    t+=1
    print('t={}, lb={},lr={},loss={}'.format(t,lb,lr,curr_loss))
    lb,lr=compute_adaptive_lambdas(model)
    if t>1e4:
        break

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(grid1[:,0].reshape(-1).detach().cpu().numpy(),
                grid1[:,1].reshape(-1).detach().cpu().numpy(),
                model(grid1).reshape(-1).detach().cpu().numpy(),
                cmap=plt.cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")

plt.show()


#lambda_bound=100


#def closure():
#    #nonlocal cur_loss
#    optimizer.zero_grad()
#    loss =wave_op(model, grid)+lambda_bound*bnd_op(model)
#    loss.backward()
#    #cur_loss = loss.item()
#    return loss

#curr_loss=10e5


#t=0

#while curr_loss>1e-3:
#    loss=optimizer.step(closure)
#    curr_loss=loss.item()
#    if t % 1000==0: print("t={}, loss={}".format(t,curr_loss))
#    t+=1
#    if t>1e4:
#        break

#import matplotlib.pyplot as plt


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_trisurf(grid[:,0].reshape(-1).detach().cpu().numpy(),
#                grid[:,1].reshape(-1).detach().cpu().numpy(),
#                model(grid).reshape(-1).detach().cpu().numpy(),
#                cmap=plt.cm.jet, linewidth=0.2, alpha=1)
#ax.set_xlabel("x1")
#ax.set_ylabel("x2")

#plt.show()
