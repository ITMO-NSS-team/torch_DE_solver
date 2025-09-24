import torch
import os
import sys
import scipy


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device, check_device


def exact_solution(grid):
    """
    Computes the reference solution on the given grid by interpolating precomputed data.
    
        This method leverages a precomputed, high-fidelity solution stored in a `.mat` file.
        It then uses nearest neighbor interpolation to map this solution onto the user-provided grid.
        This provides a benchmark for evaluating the accuracy of neural network-based solvers.
        
        Args:
            grid: The grid on which to evaluate the exact solution. This grid represents the points in the domain where the solution is desired.
        
        Returns:
            torch.Tensor: The exact solution evaluated on the given grid. This serves as a reference for comparison with the neural network's approximation.
    """
    grid = grid.to('cpu').detach()
    test_data = scipy.io.loadmat(os.path.abspath(
        os.path.join(os.path.dirname( __file__ ), '../wolfram_sln/buckley_exact.mat')))
    u = torch.from_numpy(test_data['u']).reshape(-1, 1)

    # grid_test
    x = torch.from_numpy(test_data['x']).reshape(-1, 1)
    t = torch.from_numpy(test_data['t']).reshape(-1, 1)

    grid_data = torch.cat((x, t), dim=1)

    exact = scipy.interpolate.griddata(grid_data, u, grid, method='nearest').reshape(-1)

    return torch.from_numpy(exact)
    

solver_device('cuda')

m = 0.2
L = 1
Q = -0.1
Sq = 1
mu_w = 0.89e-3
mu_o = 4.62e-3
Swi0 = 0.
Sk = 1.
t_end = 1.


def experiment(grid_res, mode):
    """
    Sets up and executes a PINN-based experiment to solve the Buckley-Leverett equation.
    
        This method configures the problem domain, boundary conditions, network architecture,
        and training procedure required to approximate the solution of the Buckley-Leverett equation.
        It trains the neural network using Adam and PSO optimizers and evaluates the accuracy
        of the obtained solution. This process leverages neural networks to learn the underlying
        physics of the equation, providing an alternative to traditional numerical methods.
    
        Args:
            grid_res: Resolution of the spatial and temporal grid.
            mode: The mode for building the domain (e.g., 'NN' or 'FEM').
    
        Returns:
            torch.nn.Sequential: The trained neural network.
    """
    
    domain = Domain()

    domain.variable('x', [0, 1], grid_res, dtype='float32')
    domain.variable('t', [0, 1], grid_res, dtype='float32')

    boundaries = Conditions()

    ##initial cond
    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=Swi0)

    ##boundary cond
    boundaries.dirichlet({'x': 0, 't': [0, 1]}, value=Sk)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 1)
    )

    def k_oil(x):
        return (1-net(x))**2

    def k_water(x):
        return (net(x))**2

    def dk_water(x):
        return 2*net(x)

    def dk_oil(x):
        return -2*(1-net(x))

    def df(x):
        return (dk_water(x)*(k_water(x)+mu_w/mu_o*k_oil(x))-
                k_water(x)*(dk_water(x)+mu_w/mu_o*dk_oil(x)))/(k_water(x)+mu_w/mu_o*k_oil(x))**2

    def coef_model(x):
        return -Q/Sq*df(x)

    equation = Equation()

    buckley_eq = {
        'm*ds/dt**1':
            {
                'coeff': m,
                'ds/dt': [1],
                'pow': 1
            },
        '-Q/Sq*df*ds/dx**1':
            {
                'coeff': coef_model,
                'ds/dx': [0],
                'pow':1
            }
    }

    equation.add(buckley_eq)

    model = Model(net, domain, equation, boundaries)

    model.compile(mode, lambda_operator=1, lambda_bound=10)

    img_dir=os.path.join(os.path.dirname( __file__ ), 'Buckley_Adam_PSO_img')


    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=500,
                                        patience=5,
                                        abs_loss=1e-5,
                                        randomize_parameter=1e-5,
                                        info_string_every=1000)

    cb_plots = plot.Plots(save_every=1000, print_every=None, img_dir=img_dir)

    optimizer = Optimizer('Adam', {'lr': 1e-3})

    model.train(optimizer, 10000, save_model=False, callbacks=[cb_es, cb_plots])

    grid = domain.build(mode)

    u_exact = exact_solution(grid).to('cuda')

    u_exact = check_device(u_exact).reshape(-1)

    u_pred = check_device(net(grid)).reshape(-1)

    error_rmse = torch.sqrt(torch.sum((u_exact - u_pred)**2)) / torch.sqrt(torch.sum(u_exact**2))

    print('RMSE_adam= ', error_rmse.item())

    #################

    optimizer = Optimizer('PSO', {'pop_size': 100,
                                  'b': 0.5,
                                  'c2': 0.05,
                                  'variance': 5e-2,
                                  'c_decrease': True,
                                  'lr': 5e-3})

    cb_plots = plot.Plots(save_every=100, print_every=None, img_dir=img_dir)

    model.train(optimizer, 3000, info_string_every=100, save_model=False, callbacks=[cb_plots])

    u_pred = check_device(net(grid)).reshape(-1)

    error_rmse = torch.sqrt(torch.sum((u_exact - u_pred)**2)) / torch.sqrt(torch.sum(u_exact**2))

    print('RMSE_pso= ', error_rmse.item())

    return net

for i in range(2):
    model = experiment(20, 'autograd')

## After experiment, RMSE_adam ~ 0.23, RMSE_pso ~ 0.19 or less.