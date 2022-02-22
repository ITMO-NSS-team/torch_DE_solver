import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

sys.path.append('../')

from solver_matrix import *
import time

device = torch.device('cpu')

"""
Preparing grid

Grid is an essentially torch.Tensor of a n-D points where n is the problem
dimensionality
"""

t = torch.from_numpy(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))

grid = [t]
grid = np.meshgrid(*grid)
grid = torch.tensor(grid, device=device)

"""
Preparing boundary conditions (BC)

For every boundary we define three items

bnd=torch.Tensor of a boundary n-D points where n is the problem
dimensionality

bop=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

NB! dictionary keys at the current time serve only for user-frienly 
description/comments and are not used in model directly thus order of
items must be preserved as (coeff,op,pow)

term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}

Meaning c1*u*d2u/dx2 has the form

{'coefficient':c1,
 'u*d2u/dx2': [[None],[0,0]],
 'pow':[1,1]}

None is for function without derivatives


bval=torch.Tensor prescribed values at every point in the boundary
"""

# point t=0
bnd1 = torch.from_numpy(np.array([[0]], dtype=np.float64))

bop1 = None

#  So u(0)=-1/2
bndval1 = torch.from_numpy(np.array([[-1 / 2]], dtype=np.float64))

# point t=1
bnd2 = torch.from_numpy(np.array([[1]], dtype=np.float64))

# d/dt
bop2 = {
    '1*du/dt**1':
        {
            'coefficient': 1,
            'du/dt': [0],
            'pow': 1
        }
}
bop2 = [1, [[0]], [1]]
# So, du/dt |_{x=1}=3
bndval2 = torch.from_numpy(np.array([[3]], dtype=np.float64))

# Putting all bconds together
bconds = [[bnd1, bop1, bndval1], [bnd2, [bop2], bndval2]]

"""
Defining Legendre polynomials generating equations

Operator has the form

op=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

NB! dictionary keys at the current time serve only for user-frienly 
description/comments and are not used in model directly thus order of
items must be preserved as (coeff,op,pow)



term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}

c1 may be integer, function of grid or tensor of dimension of grid

Meaning c1*u*d2u/dx2 has the form

{'coefficient':c1,
 'u*d2u/dx2': [[None],[0,0]],
 'pow':[1,1]}

None is for function without derivatives


"""


# 1-t^2
def c1(grid):
    return 1 - grid ** 2


# -2t
def c2(grid):
    return -2 * grid

def c3(n):
    return n*(n-1)

n=3

# operator is  (1-t^2)*d2u/dt2-2t*du/dt+n*(n-1)*u=0 (n=3)
legendre_poly= {
    '(1-t^2)*d2u/dt2**1':
        {
            'coeff': c1(grid), #coefficient is a torch.Tensor
            'du/dt': [0, 0],
            'pow': 1
        },
    '-2t*du/dt**1':
        {
            'coeff': c2(grid),
            'u*du/dx': [0],
            'pow':1
        },
    'n*(n-1)*u**1':
        {
            'coeff': 6,
            'u':  [None],
            'pow': 1
        }
}

# this one is to show that coefficients may be a function of grid as well
# legendre_poly= {
#     '(1-t^2)*d2u/dt2**1':
#         {
#             'coeff': c1, #coefficient is a function
#             'du/dt': [0, 0],
#             'pow': 1
#         },
#     '-2t*du/dt**1':
#         {
#             'coeff': c2,
#             'u*du/dx': [0],
#             'pow':1
#         },
#     'n*(n-1)*u**1':
#         {
#             'coeff': 6,
#             'u':  [None],
#             'pow': 1
#         }
# }



for _ in range(1):
    model = torch.rand(grid.shape)

    # initial_approximation = []
    # with open('initial_approximation_ODE.txt', 'r') as f:
    #     for line in f:
    #         for val in line.split():
    #             initial_approximation.append(float(val))
    #
    # model = torch.tensor(initial_approximation)

    start = time.time()
    model = lbfgs_solution(model, grid, legendre_poly, 100, bconds)
    end = time.time()

    fig = plt.figure()
    plt.scatter(grid.reshape(-1), model.reshape(-1))
    plt.scatter(grid.reshape(-1), 1 / 2 * (-1 + 3 * grid ** 2).reshape(-1))
    plt.show()

    print('Time taken 10= ', end - start)
    print(model)
