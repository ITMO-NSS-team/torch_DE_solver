import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))


from solver import *
from cache import *
from metrics import *
from input_preprocessing import *
import time
device = torch.device('cpu')
# Grid
x_grid = np.linspace(0,1,21)
t_grid = np.linspace(0,1,21)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()

grid.to(device)
# Boundary and initial conditions

# u(x,0)=1e4*sin^2(x(x-1)/10)

func_bnd1 = lambda x: 10 ** 4 * np.sin((1/10) * x * (x-1)) ** 2
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bndval1 = func_bnd1(bnd1[:,0])

# du/dx (x,0) = 1e3*sin^2(x(x-1)/10)
func_bnd2 = lambda x: 10 ** 3 * np.sin((1/10) * x * (x-1)) ** 2
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bop2 = {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            }
}
bndval2 = func_bnd2(bnd2[:,0])

# u(0,t) = u(1,t)
bnd3_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)),t).float()
bnd3_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)),t).float()
bnd3 = [bnd3_left,bnd3_right]

# du/dt(0,t) = du/dt(1,t)
bnd4_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)),t).float()
bnd4_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)),t).float()
bnd4 = [bnd4_left,bnd4_right]

bop4= {
        'du/dt':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            }
}
bcond_type = 'periodic'

bconds = [[bnd1,bndval1],[bnd2,bop2,bndval2],[bnd3,bcond_type],[bnd4,bop4,bcond_type]]
# bconds = [[bnd1,bndval1],[bnd2,bop2,bndval2],[bnd3,bcond_type]]
# wave equation is d2u/dt2-(1/4)*d2u/dx2=0
C = 4
wave_eq = {
    'd2u/dt2':
        {
            'coeff': 1,
            'd2u/dt2': [1, 1],
            'pow': 1,
            'var': 0
        },
        '-1/C*d2u/dx2':
        {
            'coeff': -1/C,
            'd2u/dx2': [0, 0],
            'pow': 1,
            'var': 0
        }
}

# NN
model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1))

unified_bconds = bnd_unify(bconds)
prepared_grid,grid_dict,point_type = grid_prepare(grid)
full_prepared_operator = operator_prepare(wave_eq, grid_dict, subset=['central'], true_grid=grid, h=0.001)
prepared_bconds = bnd_prepare(bconds,grid,grid_dict,h=0.001)


lambda_bound=10
norm=None
op = apply_operator_set(model, full_prepared_operator)
if bconds == None:
    loss = torch.mean((op) ** 2)

true_b_val_list = []
b_val_list = []
b_pos_list = []
residual = []
b_val_temp = []
# we apply no  boundary conditions operators if they are all None

# проходим по каждому гр. условию
for bcond in prepared_bconds:

    b_pos = bcond[0] # Расположение для одной границы на сетке
    b_cond_operator = bcond[1] # Оператор на границе
    if len(bcond[2]) == bcond[2].shape[-1]:  # значения на границе
        true_boundary_val = bcond[2].reshape(-1, 1)
    else:
        true_boundary_val = bcond[2]
    bnd_type = bcond[3] # тип граничных условий
    # Расчет для простых гр условий
    if bnd_type == 'boundary values':
        b_pos_list.append(bcond[0]) # составляем общий список с расположением точек на сетке для вычисления lp norm
        # проверяем на наличие оператора на границе и считаем модель на границе
        if b_cond_operator == None or b_cond_operator == [[1, [None], 1]]:
            b_op_val = model(grid)
        else:
            b_op_val = apply_operator_set(model, b_cond_operator)
        # записываем в список остатков разницу между значением модели (с учетом расположения точек на границе) и значением на границе
        residual.append(b_op_val[b_pos] - true_boundary_val)


    # периодические условия
    if bnd_type == 'periodic':
        b_pos_list.append(flatten_list(bcond[0])) # составляем общий список с расположением точек на сетке для вычисления lp norm
        # проверяем на наличие оператора на границе и считаем модель на границе
        if b_cond_operator == None or b_cond_operator == [[1, [None], 1]]:
            b_op_val = model(grid)
            # print('Значения на сетке без оператора:', b_op_val)
        else:
            b_op_val = apply_operator_set(model, b_cond_operator)
            # print('Значения на сетке с оператором:', b_op_val)

        # print('Расположение точек на границе',b_pos)
        print('Точки слева',b_op_val[b_pos[0]])
        print('Точки справа',b_op_val[b_pos[1]])
        # считаем разницу на границах слева и справа и записываем в список остатков
        residual.append(b_op_val[b_pos[0]] - b_op_val[b_pos[1]])
        print('остатки периодич',b_op_val[b_pos[0]] - b_op_val[b_pos[1]])
        # print('все остатки',residual)
        # считаем разницу на границах справа и слева и записываем в общий список остатков
# print('остатки',residual)
residual = torch.cat(residual)
# print(residual)

if norm == None:
    op_weigthed = False
    op_normalized = False
    op_p = 2
    b_weigthed = False
    b_normalized = False
    b_p = 2
else:
    op_weigthed = norm['operator_weighted']
    op_normalized = norm['operator_normalized']
    op_p = norm['operator_p']
    b_weigthed = norm['boundary_weighted']
    b_normalized = norm['boundary_weighted']
    b_p = norm['boundary_p']

loss = lp_norm(grid[:len(op)], op, weighted=op_weigthed, normalized=op_normalized, p=op_p) + \
       lambda_bound * lp_norm(grid[flatten_list(b_pos_list)], residual, p=b_p, weighted=b_weigthed,
                              normalized=b_normalized)
print(loss)