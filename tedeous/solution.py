from __future__ import annotations

import torch
import numpy as np
from copy import deepcopy
from typing import Tuple, Union

from tedeous.points_type import Points_type
from tedeous.derivative import Derivative
from tedeous.eval import Operator, Bounds
from tedeous.losses import Losses
from tedeous.device import device_type, check_device
import tedeous.input_preprocessing
from tedeous.utils import *

flatten_list = lambda t: [item for sublist in t for item in sublist]

class Solution():
    """
    class for different loss functions calculation.
    """
    def __init__(self, grid: torch.Tensor, equal_cls: Union[tedeous.input_preprocessing.Equation_NN,
                                                            tedeous.input_preprocessing.Equation_mat,
                                                            tedeous.input_preprocessing.Equation_autograd],
                 model: Union[torch.nn.Sequential, torch.Tensor], mode: str, weak_form, lambda_operator, lambda_bound):

        self.grid = check_device(grid)
        if mode == 'NN':
            sorted_grid = Points_type(self.grid).grid_sort()
            self.n_t = len(sorted_grid['central'][:, 0].unique())
        elif mode == 'autograd':
            self.n_t = len(self.grid[:, 0].unique())
        elif mode == 'mat':
            self.n_t = grid.shape[1]
        equal_copy = deepcopy(equal_cls)
        self.prepared_operator = equal_copy.operator_prepare()
        self.prepared_bconds = equal_copy.bnd_prepare()
        self.model = model.to(device_type())
        self.mode = mode
        self.weak_form = weak_form
        self.lambda_operator = lambda_operator
        self.lambda_bound = lambda_bound
        self.operator = Operator(self.grid, self.prepared_operator, self.model,
                                   self.mode, weak_form)
        self.boundary = Bounds(self.grid, self.prepared_bconds, self.model,
                                   self.mode, weak_form)

        self.op_list = []
        self.bval_list = []
        self.loss_list = []

    def evaluate(self,
                 second_order_interactions: bool = True,
                 sampling_N: int = 1,
                 lambda_update: bool = False ,
                 tol: float = 0,
                 save_graph = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes loss.

        Args:
            second_order_interactions: optimizer iteration (serves only for computing adaptive lambdas).
            sampling_N: parameter for accumulation of solutions (op, bcs). The more sampling_N, the more accurate the estimation of the variance.
            lambda_update: update lambda or not.
            tol: float constant, influences on error penalty.
            save_graph: boolean constant, responsible for saving the computational graph.

        Returns:
            loss
        """

        op = self.operator.operator_compute()
        bval, true_bval = self.boundary.apply_bcs()

        loss_cls = Losses(operator=op, bval=bval, true_bval=true_bval, lambda_op=self.lambda_operator,
                          lambda_bound=self.lambda_bound, mode=self.mode,
                          weak_form=self.weak_form, n_t=self.n_t, save_graph=save_graph)

        loss, loss_normalized = loss_cls.compute(tol)

        if lambda_update:
            # TODO refactor this lambda thing to class or function.
            op, bcs, op_length, bval_length = lambda_preproc(op, bval, true_bval)

            self.op_list.append(list_to_vector(op.values()).cpu().detach().numpy())
            self.bval_list.append(list_to_vector(bcs.values()).cpu().detach().numpy())
            self.loss_list.append(float(loss_normalized.item()))

            sampling_amount, sampling_D = samples_count(second_order_interactions = second_order_interactions,
                                                             sampling_N = sampling_N,
                                                             op_length=op_length, bval_length = bval_length)

            if len(self.op_list) == sampling_amount:
                self.lambda_operator, self.lambda_bound = Lambda(self.op_list, self.bval_list, self.loss_list,
                                                                 second_order_interactions)\
                                                                 .update(op_length=op_length,
                                                                         bval_length=bval_length,
                                                                         sampling_D=sampling_D)
                self.op_list.clear()
                self.bval_list.clear()
                self.loss_list.clear()

                lambda_print(self.lambda_operator)
                lambda_print(self.lambda_bound)


        return loss, loss_normalized
