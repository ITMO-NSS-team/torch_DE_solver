import os,sys
import torch
import numpy as np
from scipy.integrate import quad

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.device import solver_device
from landscape_visualization._aux.plot_loss_surface import PlotLossSurface


mu = 0.01 / np.pi

# model = torch.load('model.pth', map_location=torch.device('cpu'))
solver_device('gpu')

current_file_folder = os.path.abspath(os.path.dirname(__file__))
# Burgers equation problem describtion

def u(grid):
    """
    Computes the solution to a partial differential equation on a given grid using a neural network-based solver.
    
        This method leverages numerical integration with quadrature to approximate the solution at each grid point. By using a neural network, it aims to efficiently solve differential equations, providing a flexible alternative to traditional numerical methods. The solution is returned as a tensor, suitable for further analysis or visualization within the PyTorch framework.
    
        Args:
            grid: A list of tuples representing the grid points where the solution
                should be evaluated. Each tuple contains the x and t coordinates of a
                point.
    
        Returns:
            A tensor containing the solution values at each grid point, as approximated by the neural network.
    """
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
    """
    Applies a neural network to an input and detaches the result for CPU-based differential equation solving.
    
        This method ensures that the neural network and input are processed on the CPU,
        making it suitable for environments where GPU acceleration is not available or desired.
        It applies the network to the input and detaches the result from the computation graph
        to prevent unnecessary gradient calculations during the solving process. This is crucial
        for managing memory and computational resources when dealing with complex differential equations.
        
        Args:
            net: The neural network to apply.
            x: The input to the neural network.
        
        Returns:
            The output of the neural network, detached from the computation graph.
    """
    net = net.to('cpu')
    x = x.to('cpu')
    return net(x).detach()


def l2_norm(net, x):
    """
    Calculates the L2 norm between the neural network's approximation and the analytical solution. This metric quantifies the accuracy of the neural network in solving the differential equation by measuring the difference between the predicted solution and the true solution.
    
        Args:
            net: The neural network model used to approximate the solution.
            x: The input tensor representing the spatial or temporal domain of the differential equation.
    
        Returns:
            numpy.ndarray: The L2 norm, a scalar value representing the overall error, as a NumPy array.
    """
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict - exact) ** 2))
    return l2_norm.detach().cpu().numpy()


def burgers1d_problem_formulation(grid_res):
    """
    Sets up the 1D Burgers' equation problem for neural network-based solving.
    
    This method defines the domain, boundary conditions, and the Burgers' equation itself,
    preparing them for approximation using a neural network. It is a crucial step in
    formulating the problem in a way that the neural network can learn the solution.
    
    Args:
        grid_res (int): The resolution of the grid, determining the density of points
                          at which the solution will be approximated.
    
    Returns:
        tuple: A tuple containing the grid, domain, equation, and boundary conditions.
               These components are essential for training the neural network to solve
               the Burgers' equation.
            - grid (torch.Tensor): The computational grid representing the problem domain.
            - domain (Domain): The problem domain, defining the spatial and temporal extents.
            - equation (Equation): The equation to be solved, in this case, the 1D Burgers' equation.
            - boundaries (Conditions): The boundary conditions that constrain the solution space.
    """
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
    Generates lists of keys representing the available trained models, which are stored as '.pt' files within subfolders of a given directory. These keys are used to identify and load specific trained models for solving differential equations.
    
        Args:
            base_folder (str): Path to the directory containing subfolders, each representing a different trained model stored as '.pt' files.
    
        Returns:
            tuple: Two lists:
                - key_models (list): A list of strings, where each string is an index representing a specific model file across all subfolders.
                - key_modelnames (list): A list of strings, where each string is a sequential identifier (starting from "0") assigned to each model.
    
        WHY: This function creates an index of available pre-trained models, allowing the solver to load and utilize specific models for approximating solutions to differential equations. The keys generated here are essential for selecting the appropriate trained network.
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
