# this one contain some stuff for computing different auxiliary things.

import torch
import numpy as np
from SALib import ProblemSpec

def list_to_vector(list_):
    return torch.cat([x.reshape(-1) for x in list_])

def counter(fu):
    def inner(*a,**kw):
        inner.count+=1
        return fu(*a,**kw)
    inner.count = 0
    return inner
def length_dict_create(sol):
    solution_length = sol.copy()
    for key in sol:
        solution_length[key] = len(sol[key])
    return solution_length
def tensor_to_dict(tensor):
    tensor_dict = dict()
    for i in range(tensor.shape[-1]):
        tensor_dict[f'eq_{i+1}'] = tensor[:,i:i+1]
    return tensor_dict
def wrap(op, bval, true_bval):
    # op = tensor_to_dict(op)
    op_length = length_dict_create(op)
    bval_length = length_dict_create(bval)

    bcs = dict()
    for k in bval:
        bcs[k] = bval[k] - true_bval[k]

    return op, bcs, op_length, bval_length

class Lambda:
    """
    Serves for computing adaptive lambdas.
    """
    def __init__(self,op_list: list, bcs_list: list, loss_list: list,
                 sampling_N: int = 1, second_order_interactions = True):
        """

        Args:
            op_list: list with operator solution.
            bcs_list: list with boundary solution.
            loss_list: list with losses.
            sampling_N:
            second_order_interactions
        """
        self.second_order_interactions = second_order_interactions
        self.op_list = op_list
        self.bcs_list = bcs_list
        self.loss_list = loss_list
        self.sampling_N = sampling_N


    @staticmethod
    def lambda_compute(pointer: int, length_dict: dict, total_disp: float, ST: np.ndarray) -> dict:
        """

        Args:
            pointer
            length_dict
            total_disp
            ST

        Returns
        -------

        """
        lambda_dict = length_dict.copy()
        for key, value in length_dict.items():
            lambda_dict[key] = total_disp / sum(ST[pointer:pointer + value])
            pointer += value
        return lambda_dict

    def update(self,op_length, bcs_length, sampling_D):
        op_array = np.array(self.op_list)
        bc_array = np.array(self.bcs_list)
        loss_array = np.array(self.loss_list)

        X_array = np.hstack((op_array, bc_array))

        bounds = [[-100, 100] for _ in range(sampling_D)]
        names = ['x{}'.format(i) for i in range(sampling_D)]

        sp = ProblemSpec({'names': names, 'bounds': bounds, 'nprocs': 6})
        sp.set_samples(X_array)
        sp.set_results(loss_array)
        sp.analyze_sobol(calc_second_order=self.second_order_interactions)

        '''
        To assess variance we need total sensitiviy indices for every variable
        '''
        ST = sp.analysis['ST']

        '''
        Total variance is the sum of total indices
        '''
        total_disp = sum(ST)

        lambda_op = self.lambda_compute(0, op_length, total_disp, ST)
        lambda_bnd = self.lambda_compute(sum((op_length.values())), bcs_length, total_disp, ST)

        return lambda_op, lambda_bnd
