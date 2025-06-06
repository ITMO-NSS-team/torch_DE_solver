import torch
import numpy as np
from scipy.integrate import quad
from tedeous.data import Domain, Conditions, Equation
from tedeous.device import solver_device
from landscape_visualization._aux.plot_loss_surface import PlotLossSurface
import os

mu = 0.01 / np.pi

# model = torch.load('model.pth', map_location=torch.device('cpu'))
solver_device('gpu')

current_file_folder = os.path.abspath(os.path.dirname(__file__))
# Burgers equation problem describtion

def u(grid):
    def f(y):
        return np.exp(-np.cos(np.pi * y) / (2 * np.pi * mu))

    def integrand1(m, x, t):
        return np.sin(np.pi * (x - m)) * f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def integrand2(m, x, t):
        return f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def u(x, t):
        if t == 0:
            return -np.sin(np.pi * x)
        else:
            return -quad(integrand1, -np.inf, np.inf, args=(x, t))[0] / quad(integrand2, -np.inf, np.inf, args=(x, t))[
                0]

    solution = []
    for point in grid:
        solution.append(u(point[0].item(), point[1].item()))

    return torch.tensor(solution)


def u_net(net, x):
    net = net.to('cpu')
    x = x.to('cpu')
    return net(x).detach()


def l2_norm(net, x):
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict - exact) ** 2))
    return l2_norm.detach().cpu().numpy()


def burgers1d_problem_formulation(grid_res):
    domain = Domain()
    domain.variable('x', [-1, 1], grid_res)
    domain.variable('t', [0, 1], grid_res)

    boundaries = Conditions()
    x = domain.variable_dict['x']
    boundaries.dirichlet({'x': [-1, 1], 't': 0}, value=-torch.sin(np.pi * x))

    boundaries.dirichlet({'x': -1, 't': [0, 1]}, value=0)

    boundaries.dirichlet({'x': 1, 't': [0, 1]}, value=0)

    equation = Equation()

    burgers_eq = {
        'du/dt**1':
            {
                'coeff': 1.,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            },
        '+u*du/dx':
            {
                'coeff': 1,
                'u*du/dx': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 0]
            },
        '-mu*d2u/dx2':
            {
                'coeff': -mu,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(burgers_eq)

    grid = domain.build('autograd')

    return grid, domain, equation, boundaries


def generate_key_lists(base_folder):
    """
    Generate key_models and key_modelnames lists based on .pt files in subfolders.
    
    Args:
        base_folder (str): Path to the folder containing subfolders with .pt files.
    
    Returns:
        tuple: Two lists:
            - key_models: Indices of model files across all subfolders.
            - key_modelnames: Sequential names starting from 0.
    """
    key_models = []
    key_modelnames = []
    num_files = 0
    if not any(os.path.isdir(os.path.join(base_folder, item)) for item in os.listdir(base_folder)):
        return ["0"], ["0"]
    for subfolder in sorted(os.listdir(base_folder)):
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.isdir(subfolder_path):
            key_models.append(str(num_files))
            pt_files = [f for f in os.listdir(subfolder_path) if f.endswith('.pt')]
            num_files = num_files + len(pt_files)
            key_modelnames.append(str(len(key_modelnames)))

    return key_models, key_modelnames


if __name__ == '__main__':

    path_to_trajectories = os.path.join(current_file_folder, "trajectories", "burgers", "adam_5_starts")
    path_to_model = os.path.join(current_file_folder, "saved_models", "PINN_burgers_adam_5_starts", "model.pt") # Replace with the path to your folder with  models in it if you needed it
    key_models, key_modelnames = generate_key_lists(path_to_trajectories)

    model_layers = [2, 32, 32, 1]  # PINN layers
    grid_res = 80
    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
    u_exact_test = u(grid_test).reshape(-1)
    plot_args = {
        "loss_types": ["loss_total"],
        "every_nth": 1,
        "num_of_layers": 3,
        "layers_AE": [
            991,
            125,
            15
        ],
        "batch_size": 32,
        "path_to_plot_model": path_to_model, # Replace with the path to your model if you needed it. path like path_to_folder + model.pt
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "path_to_trajectories": path_to_trajectories,
        "loss_name": "train_loss",
        "x_range": [-1.25, 1.25, 25],
        "vmax": -1.0,
        "vmin": -1.0,
        "vlevel": 30.0,
        "key_models": key_models,
        "key_modelnames": key_modelnames,
        "density_type": "CKA",
        "density_p": 2,
        "density_vmax": -1,
        "density_vmin": -1,
        "colorFromGridOnly": True
    }

    plotter = PlotLossSurface(**plot_args)
    grid, domain, equation, boundaries = burgers1d_problem_formulation(grid_res)
    plotter.plotting_equation_loss_surface(u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers)
