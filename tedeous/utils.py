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
    """ Create random tensors to add some variance to torch neural network.

    Args:
        eps (float): randomize parameter.

    Returns:
        callable: creating random params function.
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
    """ Count samples for variance based sensitivity analysis.

    Args:
        second_order_interactions (bool): Calculate second-order sensitivities.
        sampling_N (int): essentially determines how often the lambda will be re-evaluated.
        op_length (list): operator values length.
        bval_length (list): boundary value length.

    Returns:
        sampling_amount (int): overall sampling value.
        sampling_D (int): sum of length of grid and boundaries.
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
    """ Print lambda value.

    Args:
        lam (torch.Tensor): lambdas values.
        keys (List): types of lambdas.
    """

    lam = lam.reshape(-1)
    for val, key in zip(lam, keys):
        print('lambda_{}: {}'.format(key, val.item()))


def bcs_reshape(
        bval: torch.Tensor,
        true_bval: torch.Tensor,
        bval_length: List) -> Tuple[dict, dict, dict, dict]:
    """ Preprocessing for lambda evaluating.

    Args:
        bval (torch.Tensor): matrix, where each column is predicted
                      boundary values of one boundary type.
        true_bval (torch.Tensor): matrix, where each column is true
                            boundary values of one boundary type.
        bval_length (list): list of length of each boundary type column.

    Returns:
        torch.Tensor: vector of difference between bval and true_bval.
    """

    bval_diff = bval - true_bval

    bcs = torch.cat([bval_diff[0:bval_length[i], i].reshape(-1)
                     for i in range(bval_diff.shape[-1])])

    return bcs


def remove_all_files(folder: str) -> None:
    """ Remove all files from folder.

    Args:
        folder (str): folder name.
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
    """ Preparation of coefficients in the operator of the *mat* method
        to suit methods *NN, autograd*.

    Args:
        operator (dict): operator (equation dict).

    Returns:
        operator (dict): operator (equation dict) with suitable coefficients.
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
    """ Create model for *NN or autograd* modes from grid
        and model of *mat* mode.

    Args:
        model (torch.Tensor): model from *mat* method.
        grid (torch.Tensor): grid from *mat* method.
        cache_model (torch.nn.Module, optional): neural network that will
                                                    approximate *mat* model. Defaults to None.

    Returns:
        cache_model (torch.nn.Module): model satisfying the *NN, autograd* methods.
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
    Saves model in a cache (uses for 'NN' and 'autograd' methods).
    Args:
        cache_dir (str): path to cache folder.
        model (torch.nn.Module): model to save.
        (uses only with mixed precision and device=cuda). Defaults to None.
        name (str, optional): name for a model. Defaults to None.
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
    """ Saves model in a cache (uses for 'mat' method).

    Args:
        cache_dir (str): path to cache folder.
        model (torch.Tensor): *mat* model
        grid (torch.Tensor): grid from *mat* mode
        cache_model (Union[torch.nn.Module, None], optional): model to save. Defaults to None.
        name (Union[str, None], optional): name for a model. Defaults to None.
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
    """ Make tensor from tuple (or None element) ad replace None elements to zero.

    Args:
        tuple_data (tuple): path to cache folder.
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
    """Pad tensor to a fixed length with given padding value.

    src: https://pytorch.org/text/stable/transforms.html#torchtext.transforms.PadTransform

    Done to avoid torchtext dependency (we need only this function).
    """

    def __init__(self, max_length: int, pad_value: int) -> None:
        """_summary_

        Args:
            max_length (int): Maximum length to pad to.
            pad_value (int):  Value to pad the tensor with.
        """
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Tensor padding

        Args:
            x (torch.Tensor): tensor for padding.

        Returns:
            torch.Tensor: filled tensor with pad value.
        """

        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x


def load_data(datapath: str) -> torch.Tensor:
    """
    Loads data from a given file and returns it as a PyTorch tensor.
    This function supports loading data from .dat and .npy files.
    If the file format is unsupported, an error is raised.

    Args:
        datapath (str): Path to the data file. Supports .dat (text format) and .npy (NumPy binary format).

    Returns:
        torch.Tensor: The loaded data as a PyTorch tensor.

    Raises:
        ValueError: If the file format is not .dat or .npy.
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
    Loads exact solution data and interpolates it onto a grid.

    Args:
        grid (torch.Tensor): the coordinate grid on which the solution will be interpolated.
        datapath (str): path to the file containing exact solution data.
        pde_dim_in (int): number of input variables for the differential equation.
        pde_dim_out (int): number of output variables (solution dimensionality).
        t_dim_flag (bool): flag indicating whether there is a time component in the data.
                                     Set to True if time is included. Defaults to False.

    Returns:
        torch.Tensor: the interpolated exact solution, with shape (N, pde_dim_out) for multidimensional
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
    Loads initial condition data and interpolates it onto the specified grid.

    Args:
        grid (torch.Tensor): coordinate grid where the initial values will be interpolated.
        datapath (str): path to the file containing initial condition data.

    Returns:
        torch.Tensor: interpolated initial condition values on the grid, shape (N,).
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
