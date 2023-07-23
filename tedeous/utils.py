# this one contain some stuff for computing different auxiliary things.

import torch
import numpy as np
from typing import Tuple

from SALib import ProblemSpec

def list_to_vector(list_):
    return torch.cat([x.reshape(-1) for x in list_])

def counter(fu):
    def inner(*a, **kw):
        inner.count += 1
        return fu(*a, **kw)

    inner.count = 0
    return inner

def tensor_to_dict(keys):
    """
        Convert nested tensor to dict.

        Args:
            tensor: nested tensor which needs to be converted into a dictionary.

        Returns:
            dictionary with tensor values.

        Examples:
            Converts operator solution for unified notation.
            Keys are number of equation, value is tensor with corresponding solution.

            >>> op = tedeous.eval.Operator(grid, prepared_operator, model, mode, weak_form).operator_compute()
            >>> tensor_to_dict(op, 'eq')
            Output: op_dict
        """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            tensor = func(self, *args, **kwargs)
            dictionary = {}
            for i in range(tensor.shape[-1]):
                dictionary[keys + f'_{i+1}'] = tensor[:,i:i+1]
            return dictionary
        return wrapper
    return decorator

def length_dict_create(dict_: dict) -> dict:
    """
    Counts the length of each value in a dictionary.

    Args:
        dict_: source dictionary.

    Returns:
        dict where values are lengths.
    """
    solution_length = dict_.copy()
    for key in dict_:
        solution_length[key] = len(dict_[key])
    return solution_length

def samples_count(second_order_interactions: bool,
                  sampling_N: int,
                  op_length: dict,
                  bval_length:dict) -> Tuple[int, int]:
    """
    Count samples for variance based sensitivity analysis.

    Args:
        second_order_interactions:
        sampling_N: essentially determines how often the lambda will be re-evaluated.
        op_length: operator values length.
        bval_length: boundary value length.

    Returns:
        sampling_amount: overall sampling value.
        sampling_D: sum of length of grid and boundaries.


    """
    grid_len = sum(op_length.values())
    bval_len = sum(bval_length.values())

    sampling_D = grid_len + bval_len

    if second_order_interactions:
        sampling_amount = sampling_N * (2 * sampling_D + 2)
    else:
        sampling_amount = sampling_N * (sampling_D + 2)
    return sampling_amount, sampling_D

def lambda_print(dict_: dict) -> None:
    """
    Print lambda value.

    Args:
        dict_: dict with lambdas.
    """
    for key in dict_.keys():
        print('lambda_{}: {}'.format(key, dict_[key]))

def lambda_preproc(op: dict,
         bval: dict,
         true_bval: dict) -> Tuple[dict, dict, dict, dict]:
    """
    Preprocessing for lambda evaluating.

    Args:
        op: dict with operator solution.
        bval: dict with boundary solution.
        true_bval: dict with true boundary solution (i.e. right side of equation).

    Returns:
        op: dict with operator solution.
        bcs: dict with difference of bval and true_bval.
        op_length: dict with lengths of operator solution.
        bval_length: dict with lengths of boundary solution.
    """

    op_length = length_dict_create(op)
    bcs_length = length_dict_create(bval)

    bcs = dict()
    for k in bval:
        bcs[k] = bval[k] - true_bval[k]

    return op, bcs, op_length, bcs_length

class Lambda:
    """
    Serves for computing adaptive lambdas.
    """
    def __init__(self, op_list: list,
                 bcs_list: list,
                 loss_list: list,
                 sampling_N: int = 1,
                 second_order_interactions = True):
        """
        Args:
            op_list: list with operator solution.
            bcs_list: list with boundary solution.
            loss_list: list with losses.
            sampling_N: parameter for accumulation of solutions (op, bcs). The more sampling_N, the more accurate the estimation of the variance.
            second_order_interactions: computes second order Sobol indices.
        """
        self.second_order_interactions = second_order_interactions
        self.op_list = op_list
        self.bcs_list = bcs_list
        self.loss_list = loss_list
        self.sampling_N = sampling_N

    @staticmethod
    def lambda_compute(pointer: int, length_dict: dict, ST: np.ndarray) -> dict:
        """
        Computes lambdas.

        Args:
            pointer: the label to calculate the lambda for the corresponding parameter.
            length_dict: dict where values are lengths.
            ST: result of SALib.ProblemSpec().

        Returns:
            dict with lambdas.

        """
        lambda_dict = length_dict.copy()
        for key, value in length_dict.items():
            lambda_dict[key] = sum(ST) / sum(ST[pointer:pointer + value])
            pointer += value
        return lambda_dict

    def update(self, op_length: dict,
               bval_length: dict,
               sampling_D: int) -> Tuple[dict, dict]:
        """
        Updates all lambdas (operator and boundary).

        Args:
            op_length: dict with lengths of operator solution.
            bval_length: dict with lengths of boundary solution.
            sampling_D: sum of op_length and bval_length.

        Returns:
            lambda_operator: values of lambdas for operator.
            lambda_bound: values of lambdas for boundary.
        """
        op_array = np.array(self.op_list)
        bc_array = np.array(self.bcs_list)
        loss_array = np.array(self.loss_list)

        X_array = np.hstack((op_array, bc_array))

        bounds = [[-100, 100] for _ in range(sampling_D)]
        names = ['x{}'.format(i) for i in range(sampling_D)]

        sp = ProblemSpec({'names': names, 'bounds': bounds})
        sp.set_samples(X_array)
        sp.set_results(loss_array)
        sp.analyze_sobol(calc_second_order=self.second_order_interactions)

        '''
        To assess variance we need total sensitiviy indices for every variable
        '''
        ST = sp.analysis['ST']

        lambda_op = self.lambda_compute(0, op_length, ST)

        lambda_bnd = self.lambda_compute(sum((op_length.values())), bval_length, ST)

        return lambda_op, lambda_bnd
