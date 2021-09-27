# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 08:26:39 2020

@author: Sashka
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from scipy.interpolate import interpn
from scipy.optimize import minimize
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
# import tensorflow as tf
import torch
from derivative import derivative

# tf.config.set_visible_devices([], 'GPU')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def advanced_riffling(a, b):
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c


# @numba.jit
def take_derivative(u, grid, dif_type, scheme_order=1, boundary_order=2):
    const, var, order, power = dif_type
    var = 1 - var
    h = grid[var]
    der = u
    for i in range(order):
        # der = np.gradient(der, h, axis=var, edge_order=1)
        der = derivative(der, h, var, scheme_order=scheme_order, boundary_order=boundary_order)
        # der = derivative_4p(der, h, var)
    der = const * der ** power
    return der


def apply_const_operator(u, grid, operator, scheme_order=1, boundary_order=2):
    total = 0
    for term in operator:
        up = u
        for dif in term:
            up = take_derivative(up, grid, dif, scheme_order=scheme_order, boundary_order=boundary_order)
        total += up
    return total


def string_reshape(string, grid):
    shapes = [len(grid[i]) for i in range(len(grid))]
    x = len(grid[0])
    y = len(grid[1])
    u = string.reshape(shapes)
    return u


def operator_norm(u, grid, operator, norm_lambda, bcond, scheme_order=1, boundary_order=2) -> float:
    op = apply_const_operator(u, grid, operator, scheme_order=scheme_order, boundary_order=boundary_order)
    bond = []
    for condition in bcond:
        bond_op = u
        if 'operator' in condition:
            bond_op = apply_const_operator(u, grid, condition['operator'])

        bond_part = torch.flatten(torch.index_select(bond_op, condition['axis'], torch.tensor(condition['boundary'], device=device)))
        bond.append(torch.linalg.norm(bond_part - condition['string']))

    if torch.allclose(u / torch.max(u), torch.zeros_like(u) + 1):
        norm = torch.tensor(1e10, requires_grad=True)
    else:
        norm = torch.linalg.norm(op) + norm_lambda * sum(bond)
    return norm

def lbfgs_solution(u, grid, operator, norm_lambda, bcond, scheme_order=1, boundary_order=2):
    operator_norm_1 = lambda x: operator_norm(x,
                                              grid,
                                              operator,
                                              norm_lambda,
                                              bcond,
                                              scheme_order=scheme_order,
                                              boundary_order=boundary_order)

    cur_loss = float('inf')
    tol = 1e-15

    optimizer = torch.optim.LBFGS([u],
                                  # lr=1e-3,
                                  # tolerance_change=5e-15,
                                  max_iter=3000,
                                  # tolerance_grad=1e-10,
                                  line_search_fn="strong_wolfe",
                                  history_size=200,
                                  )

    for i in range(10000):
        past_loss = cur_loss

        def closure():
            nonlocal cur_loss
            optimizer.zero_grad()
            loss = operator_norm_1(u)
            loss.backward()
            cur_loss = loss.item()
            return loss

        optimizer.step(closure)

        if abs(cur_loss - past_loss) / abs(cur_loss) < tol:
            print("number of steps ", i)
            break

    return u

def solution_interp(grid, field, new_grid):
    values = []
    for i in range(len(new_grid[0])):
        for j in range(len(new_grid[1])):
            values.append(interpn(grid, field, (new_grid[0][i], new_grid[1][j]))[0])
    values = np.array(values)
    values = values.reshape([len(new_grid[0]), len(new_grid[1])])
    return values


def solution_interp_RBF(grid, field, new_grid, method='multiquadric', smooth=0):
    mesh = np.meshgrid(*grid)
    full_list = [mesh[i] for i in range(len(mesh))]
    full_list.append(field)
    interp = Rbf(*full_list, method=method, smooth=smooth)
    new_mesh = np.meshgrid(*new_grid)
    values = interp(*new_mesh)
    return values


# # demonstrate data normalization with sklearn
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# # Dependencies
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout

# from keras.callbacks import EarlyStopping


def solution_interp_nn(grid, matrix, new_grid):
    matrix = np.array(matrix)
    X, T = np.meshgrid(grid[1], grid[0])
    grid_nn = np.array([X.reshape(-1), T.reshape(-1)]).T
    u_nn = matrix.reshape(-1)
    # create scaler
    scaler = MinMaxScaler()
    # fit scaler on data
    scaler.fit(u_nn.reshape(-1, 1))
    # apply transform
    normalized = scaler.transform(u_nn.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(grid_nn, normalized, test_size=0.1)
    # Neural network
    model = Sequential()
    model = Sequential()
    model.add(Dense(256, input_dim=2, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(loss="mean_squared_error", optimizer='adam')
    Callback = EarlyStopping(monitor='loss', min_delta=0, patience=20)
    # Callback=EarlyStopping(monitor='loss', min_delta=0, patience=300)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10000000000, batch_size=128,
                        verbose=0, callbacks=[Callback])

    X, T = np.meshgrid(new_grid[1], new_grid[0])

    grid_interp = np.array([X.reshape(-1), T.reshape(-1)]).T

    u_interp = []

    u_interp = model.predict(grid_interp)

    # inverse transform
    inverse = scaler.inverse_transform(u_interp.reshape(-1, 1))
    inverse = inverse.reshape([len(new_grid[0]), len(new_grid[1])])
    if np.allclose(inverse / np.max(inverse), np.zeros_like(inverse) + 1):
        inverse = np.random.random((len(new_grid[0]), len(new_grid[1])))
    return inverse


def rotation_matrix(n, i, j, theta):
    mat = np.identity(n)
    mat[i, i] = 0
    mat[j, j] = 0
    c = np.cos(theta)
    s = np.sin(theta)
    mat[i, i] = c
    mat[j, j] = c
    mat[i, j] = -s
    mat[j, i] = s
    return mat


def operator_norm_diag(s0, i, s, u, vh, grid, operator, op_lambda):
    if i is not None: s[i] = s0
    norm = operator_norm(np.dot(u, np.dot(np.diag(s), vh)), grid, operator, op_lambda) + 1 / np.linalg.norm(s)
    return norm


def optimize_diag(i, u, s, vh, grid, operator, op_lambda):
    if i < len(s) - 1:
        bnds = ((s[i + 1], None),)
    else:
        bnds = ((None, None),)
    opt = minimize(operator_norm_diag, s[i], args=(i, s, u, vh, grid, operator, op_lambda), bounds=bnds)
    s[i] = opt.x
    min_norm = opt.fun - 1 / np.linalg.norm(s)
    return s, min_norm


def optimize_singular_value(matrix, i, grid, operator, operator_lambda):
    u, s_opt, vh = np.linalg.svd(matrix, full_matrices=False)
    s_opt, min_norm = optimize_diag(i, u, s_opt, vh, grid, operator, operator_lambda)
    opt_matrix = np.dot(u, np.dot(np.diag(s_opt), vh))
    return opt_matrix, min_norm


def optimize_SVD_rotations_2D(matrix, align, grid, operator, operator_lambda):
    u1, s_opt, vh1 = np.linalg.svd(matrix, full_matrices=False)
    norm_list = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    if align == 'left':
        convergence = False
        while not convergence:
            j = np.random.randint(1, len(u1))
            i = np.random.randint(j)
            initial_rotation = 0
            opt = minimize(lambda grade: operator_norm(
                np.dot(np.dot(u1, rotation_matrix(len(u1), i, j, np.radians(grade))), np.dot(np.diag(s_opt), vh1)),
                grid, operator, operator_lambda), initial_rotation)
            if np.abs(opt.x) < 0.005: opt.x = 0
            rot = rotation_matrix(len(u1), i, j, np.radians(opt.x))
            u1 = np.dot(u1, rot)
            u1, s_opt, vh1 = np.linalg.svd(np.dot(u1, np.dot(np.diag(s_opt), vh1)), full_matrices=False)
            norm_list = np.roll(norm_list, -1)
            norm_list[-1] = opt.fun
            # print('rot= ',opt.x)
            # print('norm=', opt.fun)
            if np.abs(norm_list[1] - norm_list[-1]) < 0.00001:
                convergence = True
    elif align == 'right':
        convergence = False
        while not convergence:
            j = np.random.randint(1, len(vh1))
            i = np.random.randint(j)
            opt = minimize(lambda grade: operator_norm(
                np.dot(u1, np.dot(np.diag(s_opt), np.dot(vh1, rotation_matrix(len(vh1), i, j, np.radians(grade))))),
                grid, [[(1, 0, 2, 1)], [(-1 / 4, 1, 2, 1)]], 1), 0)
            if np.abs(opt.x) < 0.005: opt.x = 0
            rot = rotation_matrix(len(vh1), i, j, np.radians(opt.x))
            vh1 = np.dot(vh1, rot)
            u1, s_opt, vh1 = np.linalg.svd(np.dot(u1, np.dot(np.diag(s_opt), vh1)), full_matrices=False)
            norm_list = np.roll(norm_list, -1)
            norm_list[-1] = opt.fun
            # print('rot= ',opt.x)
            # print('norm=', opt.fun)
            if np.abs(norm_list[1] - norm_list[-1]) < 0.00001:
                convergence = True
    matrix = np.dot(u1, np.dot(np.diag(s_opt), vh1))
    min_norm = opt.fun
    return matrix, min_norm


def SVD_opt(matrix, grid, operator, operator_lambda, optimize_sing=True):
    norm_list = np.array([1, 2, 3, 4, 5], dtype=np.float64)

    min_norm = operator_norm(matrix, grid, operator, operator_lambda)

    norm_list = np.roll(norm_list, -1)
    norm_list[-1] = min_norm

    align = []
    if matrix.shape[0] == matrix.shape[1]:
        align.append('left')
        align.append('right')
    elif matrix.shape[0] < matrix.shape[1]:
        align.append('left')
    else:
        align.append('right')

    while np.abs(norm_list[1] - norm_list[-1]) > 0.005:
        if optimize_sing:
            for i in range(min(matrix.shape)):
                matrix, min_norm = optimize_singular_value(matrix, i, grid, operator, operator_lambda)
            norm_list = np.roll(norm_list, -1)
            norm_list[-1] = min_norm
        for al in align:
            matrix, min_norm = optimize_SVD_rotations_2D(matrix, al, grid, operator, operator_lambda)
            norm_list = np.roll(norm_list, -1)
            norm_list[-1] = min_norm
    return matrix


def gradient_opt(matrix, grid, operator, operator_lambda):
    norm_before = 0
    norm_after = 1
    for _ in range(20):
        opt = minimize(operator_norm, matrix.reshape(-1), args=(grid, operator, operator_lambda),
                       options={'disp': False, 'gtol': 1e-3})
        norm_before = norm_after
        norm_after = opt.fun
        matrix = opt.x
        if np.abs(norm_before - norm_after) == 0: break
    sln = string_reshape(opt.x, grid)
    return sln


def plot_3D_surface(surf, solution, grid):
    X, T = np.meshgrid(grid[1], grid[0])
    surf = surf.reshape([len(grid[0]), len(grid[1])])

    if solution is not None:
        error = np.abs(solution - surf)
        wolfram_MAE = np.mean(error)

        plt.imshow(error, cmap='coolwarm')
        plt.title('Wolfram MAE= ' + '{:.9f}'.format(wolfram_MAE))
        plt.colorbar()
        plt.show()
        plot_3D_surface(solution, None, grid)

    fig1 = plt.figure()
    ax = fig1.gca(projection='3d')
    ax.plot_surface(X, T, surf)
    plt.show()

