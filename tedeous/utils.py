"""this one contain some stuff for computing different auxiliary things."""

from typing import Tuple, List, Union, Any
from torch.nn import Module
import datetime
import os
import shutil
import numpy as np
import torch
import scipy
from tedeous.device import check_device


def create_random_fn(eps: float) -> callable:
    """
    Creates a function to inject noise into the weights and biases of linear and convolutional layers of a neural network.
    
    This helps to explore the solution space and potentially improve the network's ability to approximate the solution of a differential equation by adding slight variations to the model's parameters during training.
    
    Args:
        eps (float): The magnitude of the random noise added to the weights and biases.
    
    Returns:
        callable: A function that, when applied to a PyTorch module, adds random noise to the weights and biases of its linear and convolutional layers.
    """

    def randomize_params(m):
        if (isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d)) and m.bias is not None:
            m.weight.data = m.weight.data + \
                            (2 * torch.randn(m.weight.size()) - 1) * eps
            m.bias.data = m.bias.data + (2 * torch.randn(m.bias.size()) - 1) * eps

    return randomize_params


def samples_count(second_order_interactions: bool,
                  sampling_N: int,
                  op_length: list,
                  bval_length: list) -> Tuple[int, int]:
    """
    Calculates the number of samples required for sensitivity analysis in the neural network-based differential equation solver.
    
        This function determines the sampling requirements based on whether second-order interactions are considered and the sizes of the operator grid and boundary values.
        The sampling amount is crucial for accurately estimating the sensitivity indices, which help understand the influence of different parameters on the solution of the differential equation.
    
        Args:
            second_order_interactions (bool): A flag indicating whether to calculate second-order sensitivities, increasing the sampling requirement.
            sampling_N (int): A base sampling factor that determines the frequency of re-evaluation of the loss function.
            op_length (list): A list of integers representing the lengths of the operator values.
            bval_length (list): A list of integers representing the lengths of the boundary values.
    
        Returns:
            Tuple[int, int]: A tuple containing:
                - sampling_amount (int): The total number of samples required for the sensitivity analysis.
                - sampling_D (int): The sum of the lengths of the operator grid and boundaries, representing the total number of parameters.
    """

    grid_len = sum(op_length)
    bval_len = sum(bval_length)

    sampling_D = grid_len + bval_len

    if second_order_interactions:
        sampling_amount = sampling_N * (2 * sampling_D + 2)
    else:
        sampling_amount = sampling_N * (sampling_D + 2)
    return sampling_amount, sampling_D


def lambda_print(lam: torch.Tensor, keys: List) -> None:
    """
    Print the values of the learned parameters (lambdas) associated with different components of the differential equation.
    
    This function is used to inspect the learned weights that determine the contribution of each term in the neural network's approximation of the differential equation's solution. By examining these values, one can gain insights into the model's behavior and the relative importance of different terms in the equation.
    
    Args:
        lam (torch.Tensor): A tensor containing the learned lambda values.
        keys (List): A list of strings, where each string corresponds to the type or description of the corresponding lambda value.
    
    Returns:
        None: This function prints the lambda values to the console and does not return any value.
    """

    lam = lam.reshape(-1)
    for val, key in zip(lam, keys):
        print('lambda_{}: {}'.format(key, val.item()))


def bcs_reshape(
        bval: torch.Tensor,
        true_bval: torch.Tensor,
        bval_length: List) -> Tuple[dict, dict, dict, dict]:
    """
    Reshapes and concatenates boundary condition differences for efficient evaluation within the neural differential equation solver. This preprocessing step prepares the boundary condition data for calculating the loss function, ensuring that the predicted boundary values align with the true values during the training process.
    
        Args:
            bval (torch.Tensor): Predicted boundary values, where each column represents a different boundary type.
            true_bval (torch.Tensor): True boundary values, corresponding to the predicted values.
            bval_length (List): A list containing the length of each boundary type column.
    
        Returns:
            torch.Tensor: A concatenated vector of the differences between predicted and true boundary values, reshaped for loss calculation.
    """

    bval_diff = bval - true_bval

    bcs = torch.cat([bval_diff[0:bval_length[i], i].reshape(-1)
                     for i in range(bval_diff.shape[-1])])

    return bcs


def remove_all_files(folder: str) -> None:
    """
    Remove all files and subdirectories from the specified folder.
    
    This function is used to clean up directories, ensuring a fresh start 
    for simulations or experiments by removing any existing files or 
    directories that might interfere with the process.
    
    Args:
        folder (str): The path to the folder that needs to be cleared.
    
    Returns:
        None
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def mat_op_coeff(equation: Any) -> Any:
    """
    Prepares the coefficients within the equation to ensure compatibility with neural network-based solvers. This involves reshaping tensor coefficients and issuing warnings for callable coefficients to maintain consistency during the solution process.
    
        Args:
            equation (Equation): The equation object containing the differential equation definition.
    
        Returns:
            Equation: The modified equation object with adjusted coefficients suitable for neural network solvers.
    """

    for op in equation.equation_lst:
        for label in list(op.keys()):
            term = op[label]
            if isinstance(term['coeff'], torch.Tensor):
                term['coeff'] = term['coeff'].reshape(-1, 1)
            elif callable(term['coeff']):
                print("Warning: coefficient is callable,\
                                it may lead to wrong cache item choice")
    return equation


def model_mat(model: torch.Tensor,
              domain: Any,
              cache_model: torch.nn.Module = None) -> Tuple[torch.Tensor, torch.nn.Module]:
    """
    Creates a neural network model to approximate the solution of a differential equation, leveraging a pre-computed solution on a grid.
    
        This function takes a pre-computed solution (`model`) and a corresponding grid (`domain`) and trains a neural network (`cache_model`) to approximate this solution. This allows for efficient evaluation of the solution at arbitrary points within the domain, as the neural network can be used as a surrogate model.
    
        Args:
            model (torch.Tensor): The pre-computed solution of the differential equation on the grid.
            domain (Any): The domain on which the differential equation is defined. Used to build the grid.
            cache_model (torch.nn.Module, optional): An existing neural network model to be used for approximation. If None, a default model is created. Defaults to None.
    
        Returns:
            torch.nn.Module: A trained neural network model that approximates the solution of the differential equation.
    """
    grid = domain.build('mat')
    input_model = grid.shape[0]
    output_model = model.shape[0]

    if cache_model is None:
        cache_model = torch.nn.Sequential(
            torch.nn.Linear(input_model, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, output_model)
        )

    return cache_model


def save_model_nn(
        cache_dir: str,
        model: torch.nn.Module,
        name: Union[str, None] = None) -> None:
    """
    Saves a trained neural network model to the specified cache directory. This allows for later reuse of the trained model to approximate solutions to differential equations without retraining.
    
        Args:
            cache_dir (str): The path to the directory where the model will be saved.
            model (torch.nn.Module): The trained neural network model to be saved.
            name (str, optional): A custom name for the saved model file. If None, a timestamp-based name is generated. Defaults to None.
    
        Returns:
            None
    """

    if name is None:
        name = str(datetime.datetime.now().timestamp())
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    parameters_dict = {'model': model.to('cpu'),
                       'model_state_dict': model.state_dict()}

    try:
        torch.save(parameters_dict, cache_dir + '\\' + name + '.tar')
        print(f'model is saved in cache dir: {cache_dir}')
    except RuntimeError:
        torch.save(parameters_dict, cache_dir + '\\' + name + '.tar',
                   _use_new_zipfile_serialization=False)  # cyrillic in path
        print(f'model is saved in cache: {cache_dir}')
    except:
        print(f'Cannot save model in cache: {cache_dir}')


def save_model_mat(cache_dir: str,
                   model: torch.Tensor,
                   domain: Any,
                   cache_model: Union[torch.nn.Module, None] = None,
                   name: Union[str, None] = None) -> None:
    """
    Refines a pre-trained solution of a differential equation by training a neural network to match it.
    
    This method takes a pre-trained solution (model) and trains a neural network
    to approximate it more closely. This is useful for improving the accuracy
    and smoothness of the initial solution obtained through other methods.
    
    Args:
        cache_dir (str): Path to the directory where the refined model will be saved.
        model (torch.Tensor): The pre-trained solution (e.g., from the 'mat' method) represented as a tensor.
        domain (Any): The domain over which the differential equation is defined.
        cache_model (Union[torch.nn.Module, None], optional): An existing neural network model to use for refinement. If None, a new model is created. Defaults to None.
        name (Union[str, None], optional): A name for the saved refined model. Defaults to None.
    
    Returns:
        None: The refined model is saved to the specified cache directory.
    """

    net_autograd = model_mat(model, domain, cache_model)
    nn_grid = domain.build('autograd')
    optimizer = torch.optim.Adam(net_autograd.parameters(), lr=0.001)
    model_res = model.reshape(-1, model.shape[0])

    def closure():
        optimizer.zero_grad()
        loss = torch.mean((net_autograd(check_device(nn_grid)) - model_res) ** 2)
        loss.backward()
        return loss

    loss = np.inf
    t = 0
    while loss > 1e-5 and t < 1e5:
        loss = optimizer.step(closure)
        t += 1
        print('Interpolate from trained model t={}, loss={}'.format(
            t, loss))

    save_model_nn(cache_dir, net_autograd, name=name)


def replace_none_by_zero(tuple_data: tuple | None) -> torch.Tensor:
    """
    Converts a tuple (or None) to a PyTorch tensor, replacing any None elements with zeros. This is useful for ensuring that input data is in a consistent tensor format suitable for neural network-based differential equation solvers.
    
        Args:
            tuple_data (tuple | None | torch.Tensor): A tuple, None, or a tensor containing data.
    
        Returns:
            torch.Tensor | tuple: A PyTorch tensor with None values replaced by zeros, or a tuple of tensors if the input was a tuple.
    """
    if isinstance(tuple_data, torch.Tensor):
        tuple_data[tuple_data == None] = 0
    elif tuple_data is None:
        tuple_data = torch.tensor([0.])
    elif isinstance(tuple_data, tuple):
        new_tuple = tuple(replace_none_by_zero(item) for item in tuple_data)
        return new_tuple
    return tuple_data


class PadTransform(Module):
    """
    Pads a tensor to a specified length using a given padding value. This transform is useful for standardizing the input size for neural network models, ensuring consistent processing of variable-length sequences or data.
    
    
        src: https://pytorch.org/text/stable/transforms.html#torchtext.transforms.PadTransform
    
        Done to avoid torchtext dependency (we need only this function).
    """


    def __init__(self, max_length: int, pad_value: int) -> None:
        """
        Pads sequences to a specified maximum length.
        
                This ensures consistent input sizes for neural network models 
                used to approximate solutions of differential equations.
        
                Args:
                    max_length (int): Maximum length to pad to.
                    pad_value (int): Value to pad the tensor with.
        
                Returns:
                    None
        """
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pads a tensor to a specified maximum length.
        
                This ensures consistent input sizes for the neural network, which is crucial when solving differential equations where the length of the encoded representation may vary.
        
                Args:
                    x (torch.Tensor): The input tensor to be padded. The last dimension will be padded.
        
                Returns:
                    torch.Tensor: The padded tensor. If the input tensor's last dimension is already equal to or greater than the maximum length, the original tensor is returned.
        """

        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x


def load_data(datapath: str) -> torch.Tensor:
    """
    Loads data from a file into a PyTorch tensor for use in differential equation solving.
    
        This function reads data from either `.dat` (text) or `.npy` (NumPy binary) files,
        converting it into a format suitable for training neural network models to approximate
        solutions to differential equations. The data is expected to represent initial conditions,
        boundary conditions, or known solutions that the neural network will learn to replicate.
    
        Args:
            datapath (str): The path to the data file.  Supported formats are `.dat` and `.npy`.
    
        Returns:
            torch.Tensor: The loaded data as a PyTorch tensor, ready for use in training.
    
        Raises:
            ValueError: If the file format is not `.dat` or `.npy`. This ensures that only supported
                data formats are used, maintaining data integrity for the neural network training process.
    """
    file_format = os.path.splitext(datapath)[-1]

    if file_format == '.dat':
        data = np.loadtxt(datapath, comments="%", encoding='utf-8').astype(np.float32)
    elif file_format == '.npy':
        data = np.load(datapath).astype(np.float32)
    else:
        raise ValueError("Unsupported file format. Please provide a .dat or .npy file.")

    return torch.from_numpy(data)


def exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out, t_dim_flag=False):
    """
    Loads the exact solution and interpolates it onto a specified grid.
    
        This is done to obtain a reference solution for comparison with the neural network's approximation.
        By interpolating the exact solution onto the same grid used for evaluating the neural network,
        we can directly assess the accuracy of the neural network solver.
    
        Args:
            grid (torch.Tensor): The coordinate grid where the exact solution will be interpolated.
            datapath (str): Path to the file containing the exact solution data.
            pde_dim_in (int): Number of input variables for the differential equation.
            pde_dim_out (int): Number of output variables (solution dimensionality).
            t_dim_flag (bool): Flag indicating whether there is a time component in the data.
                                 Set to True if time is included. Defaults to False.
    
        Returns:
            torch.Tensor: The interpolated exact solution, with shape (N, pde_dim_out) for multidimensional
                          solutions, or (N,) for single-dimensional solutions.
    """

    device_origin = grid.device
    grid = grid.to('cpu').detach()

    test_data = load_data(datapath)
    grid_data = torch.stack([coord for coord in test_data[:, :pde_dim_in - t_dim_flag]])
    exact_func = test_data[:, pde_dim_in - t_dim_flag:]

    if t_dim_flag:
        N_t = int(exact_func.shape[1] / pde_dim_out)
        exact_func = exact_func.reshape(-1, pde_dim_out)
        t = torch.linspace(min(grid[:, pde_dim_in - 1]), max(grid[:, pde_dim_in - 1]), N_t) \
            .reshape(-1, 1).to('cpu').detach()
        grid_data = torch.vstack([torch.cat((coord.expand(len(t), len(coord)), t), dim=1) for coord in grid_data])

    grid_data = grid_data.cpu().numpy()
    exact_func = exact_func.cpu().numpy()
    grid = grid.cpu().numpy()

    if pde_dim_out == 1:
        exact_func = scipy.interpolate.griddata(grid_data, exact_func, grid, method='nearest').reshape(-1)
    else:
        exact_func = np.array(
            [scipy.interpolate.griddata(grid_data, exact_func[:, i_dim], grid, method='nearest').reshape(-1)
             for i_dim in range(pde_dim_out)]
        )

    exact_func = torch.from_numpy(exact_func).to(device_origin)
    return exact_func


def init_data(grid, datapath):
    """
    Loads the initial condition data from a file and interpolates it onto a specified grid.
        This is a crucial step in setting up the problem for the neural network solver,
        as it provides the initial state from which the solution will evolve.
    
        Args:
            grid (torch.Tensor): Coordinate grid where the initial values will be interpolated.
                This grid represents the spatial or temporal domain on which the solution is sought.
            datapath (str): Path to the file containing the initial condition data.
                The data file should contain the initial values at specific points, which will then be interpolated.
    
        Returns:
            torch.Tensor: Interpolated initial condition values on the grid, shape (N,).
                These values serve as the starting point for the neural network's approximation of the differential equation's solution.
    """

    device_origin = grid.device
    grid = grid.to('cpu').detach()

    init_data = load_data(datapath)
    grid_data = torch.stack([coord for coord in init_data[:, :-1]])

    init_value = init_data[:, -1:]

    grid_data = grid_data.cpu().numpy()
    init_value = init_value.cpu().numpy()
    grid = grid.cpu().numpy()

    init_value = scipy.interpolate.griddata(grid_data, init_value, grid, method='nearest').reshape(-1)
    init_value = torch.from_numpy(init_value).to(device_origin)

    return init_value
