from solver_matrix import *
import time
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

sys.path.append('../')

device = torch.device('cpu')

x = torch.from_numpy(np.linspace(0, 1, 11))
t = torch.from_numpy(np.linspace(0, 1, 11))

grid = []
grid.append(x)
grid.append(t)

grid = np.meshgrid(*grid)
grid = torch.tensor(grid, device=device)

# Initial conditions at t=0
bnd1 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), x).float()

# u(0,x)=sin(pi*x)
bndval1 = torch.sin(np.pi * bnd1[:, 1])

# Initial conditions at t=1
bnd2 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), x).float()

# u(1,x)=sin(pi*x)
bndval2 = torch.sin(np.pi * bnd2[:, 1])

# Boundary conditions at x=0
bnd3 = torch.cartesian_prod(t, torch.from_numpy(np.array([0], dtype=np.float64))).float()

# u(0,t)=0
bndval3 = torch.from_numpy(np.zeros(len(bnd3), dtype=np.float64))

# Boundary conditions at x=1
bnd4 = torch.cartesian_prod(t, torch.from_numpy(np.array([1], dtype=np.float64))).float()

# u(1,t)=0
bndval4 = torch.from_numpy(np.zeros(len(bnd4), dtype=np.float64))

# Putting all bconds together
bconds = [[bnd1, bndval1], [bnd2, bndval2], [bnd3, bndval3], [bnd4, bndval4]]

wave_eq = {
    '4*d2u/dx2**1':
        {
            'coeff': 4,
            'd2u/dx2': [0, 0],
            'pow': 1
        },
    '-d2u/dt2**1':
        {
            'coeff': -1,
            'd2u/dt2': [1,1],
            'pow':1
        }
}

for _ in range(1):
    model = grid[0]**2 + grid[1]**2

    wolfram_sln = np.genfromtxt('wave_sln_10.csv', delimiter=',')

    # solution_print(grid, torch.from_numpy(wolfram_sln))
    start = time.time()

    model = lbfgs_solution(model, grid, wave_eq, 10, bconds)

    end = time.time()

    error = mean_squared_error(model.numpy().reshape(-1), wolfram_sln.reshape(-1))
    mae = np.mean(np.abs(model.numpy().reshape(-1) - wolfram_sln.reshape(-1)))

    print('rmse', error)
    print('mae', mae)
    print('Solution\n', model)
    print('Time taken = ', end - start)

    solution_print(grid, model)