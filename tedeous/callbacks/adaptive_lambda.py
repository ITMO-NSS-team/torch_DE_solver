import numpy as np
import torch
from typing import Tuple, List
from SALib import ProblemSpec

from tedeous.callbacks.callback import Callback

class AdaptiveLambda:
    """
    Serves for computing adaptive lambdas.
    """
    def __init__(self,
                 sampling_N: int = 1,
                 second_order_interactions = True):
        """_summary_

        Args:
            op_list (list): list with operator solution.
            bcs_list (list): list with boundary solution.
            loss_list (list): list with losses.
            sampling_N (int, optional): parameter for accumulation of solutions (op, bcs).
                The more sampling_N, the more accurate the estimation of the variance.. Defaults to 1.
            second_order_interactions (bool, optional): computes second order Sobol indices. Defaults to True.
        """

        self.second_order_interactions = second_order_interactions
        self.sampling_N = sampling_N

    @staticmethod
    def lambda_compute(pointer: int, length_list: list, ST: np.ndarray) -> dict:
        """ Computes lambdas.

        Args:
            pointer (int): the label to calculate the lambda for the corresponding parameter.
            length_list (list): dict where values are lengths.
            ST (np.ndarray): result of SALib.ProblemSpec().

        Returns:
            dict: _description_
        """

        lambdas = []
        for value in length_list:
            lambdas.append(sum(ST) / sum(ST[pointer:pointer + value]))
            pointer += value
        return torch.tensor(lambdas).float().reshape(1, -1)

    def update(self, op_length: list,
               bval_length: list,
               sampling_D: int) -> Tuple[dict, dict]:
        """ Updates all lambdas (operator and boundary).

        Args:
            op_length (list): dict with lengths of operator solution.
            bval_length (list): dict with lengths of boundary solution.
            sampling_D (int): sum of op_length and bval_length.

        Returns:
            lambda_op (torch.Tensor): values of lambdas for operator.
            lambda_bound (torch.Tensor): values of lambdas for boundary.
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

        #
        # To assess variance we need total sensitiviy indices for every variable
        #
        ST = sp.analysis['ST']

        lambda_op = self.lambda_compute(0, op_length, ST)

        lambda_bnd = self.lambda_compute(sum(op_length), bval_length, ST)

        return lambda_op, lambda_bnd