import torch
import numpy as np
from typing import Tuple, Union, Any
from tedeous.input_preprocessing import lambda_prepare


class PINNLossData:
    def __init__(self, solution_cls):
        # Храним экземпляр Solution
        self.solution_cls = solution_cls

    # def __getattr__(self, name):
    #     # Делегируем вызовы методов и атрибутов к экземпляру Solution
    #     return getattr(self.solution_cls, name)

    def evaluate(self, save_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom implementation of evaluate."""

        # Используем напрямую атрибуты и методы Solution

        self.op = self.solution_cls.operator.operator_compute()
        self.bval, self.true_bval, \
        self.bval_keys, self.bval_length = self.solution_cls.boundary.apply_bcs()

        dtype = self.op.dtype
        self.lambda_operator = lambda_prepare(self.op.detach(), self.solution_cls.lambda_operator).to(dtype)
        self.lambda_bound = lambda_prepare(self.bval, self.solution_cls.lambda_bound).to(dtype)

        if self.solution_cls.mode in ('mat', 'autograd'):
            if self.bval is None:
                print('No bconds is not possible, returning infinite loss')
                return np.inf

        inputs = [self.op.detach(),
                  self.bval,
                  self.true_bval,
                  self.lambda_operator,
                  self.lambda_bound, ]

        if self.solution_cls.weak_form is not None and self.solution_cls.weak_form != []:
            loss_dict = self._weak_loss(*inputs)
        elif self.solution_cls.tol != 0:
            loss_dict = self._causal_loss(*inputs)
        else:
            loss_dict = self._default_loss(*inputs, save_graph)

        return loss_dict

    def _loss_op(self,
                 operator: torch.Tensor,
                 lambda_op: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Operator term in loss calc-n.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().

            lambda_op (torch.Tensor): regularization parameter for operator term in loss.

        Returns:
            loss_operator (torch.Tensor): operator term in loss.
            op (torch.Tensor): MSE of operator on the whole grid.
        """
        with torch.no_grad():
            if self.solution_cls.weak_form is not None and self.solution_cls.weak_form != []:
                op = operator
            else:
                op = torch.mean(operator ** 2, 0)

            loss_operator = op @ lambda_op.T
        return loss_operator, op

    def _loss_bcs(self,
                  bval: torch.Tensor,
                  true_bval: torch.Tensor,
                  lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes boundary loss for corresponding type.

        Args:
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.

        Returns:
            loss_bnd (torch.Tensor): boundary term in loss.
            bval_diff (torch.Tensor): MSE of all boundary con-s.
        """
        with torch.no_grad():
            bval_diff = torch.mean((bval - true_bval) ** 2, 0)

            loss_bnd = bval_diff @ lambda_bound.T
        return loss_bnd, bval_diff

    def _default_loss(self,
                      operator: torch.Tensor,
                      bval: torch.Tensor,
                      true_bval: torch.Tensor,
                      lambda_op: torch.Tensor,
                      lambda_bound: torch.Tensor,
                      save_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute l2 loss.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_op (torch.Tensor): regularization parameter for operator term in loss.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
            save_graph (bool, optional): saving computational graph. Defaults to True.

        Returns:
            loss (torch.Tensor): loss.
            loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
        """

        if bval is None:
            return torch.sum(torch.mean((operator) ** 2, 0))

        loss_oper, op = self._loss_op(operator, lambda_op)
        dtype = op.dtype
        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        lambda_op_normalized = lambda_prepare(operator, 1).to(dtype)
        lambda_bound_normalized = lambda_prepare(bval, 1).to(dtype)

        with torch.no_grad():
            loss_normalized = op @ lambda_op_normalized.T + \
                              bval_diff @ lambda_bound_normalized.T

        # TODO make decorator and apply it for all losses.
        if not save_graph:
            temp_loss = loss.detach()
            del loss
            torch.cuda.empty_cache()
            loss = temp_loss
        loss_dict = {
            "loss": loss,
            "loss_normalized": loss_normalized.detach(),
            "loss_oper": loss_oper.detach(),
            "loss_bnd": loss_bnd.detach(),
            "operator": operator.detach(),
            "bval_diff": bval_diff.detach()
        }
        torch.cuda.empty_cache()
        return loss_dict

    def _causal_loss(self,
                     operator: torch.Tensor,
                     bval: torch.Tensor,
                     true_bval: torch.Tensor,
                     lambda_op: torch.Tensor,
                     lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes causal loss, which is calculated with weights matrix:
        W = exp(-tol*(Loss_i)) where Loss_i is sum of the L2 loss from 0
        to t_i moment of time. This loss function should be used when one
        of the DE independent parameter is time.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_op (torch.Tensor): regularization parameter for operator term in loss.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.

        Returns:
            loss (torch.Tensor): loss.
            loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
        """

        res = torch.sum(operator ** 2, dim=1).reshape(self.n_t, -1)
        res = torch.mean(res, axis=1).reshape(self.n_t, 1)
        m = torch.triu(torch.ones((self.n_t, self.n_t), dtype=res.dtype), diagonal=1).T
        with torch.no_grad():
            w = torch.exp(- self.tol * (m @ res))

        loss_oper = torch.mean(w * res)

        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)

        loss = loss_oper + loss_bnd

        lambda_bound_normalized = lambda_prepare(bval, 1)
        with torch.no_grad():
            loss_normalized = loss_oper + \
                              lambda_bound_normalized @ bval_diff

        loss_dict = {
            "loss": loss,
            "loss_normalized": loss_normalized.detach(),
            "loss_oper": loss_oper.detach(),
            "loss_bnd": loss_bnd.detach(),
            "operator": operator.detach(),
            "bval_diff": bval_diff.detach()
        }

        return loss_dict

    def _weak_loss(self,
                   operator: torch.Tensor,
                   bval: torch.Tensor,
                   true_bval: torch.Tensor,
                   lambda_op: torch.Tensor,
                   lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Weak solution of O/PDE problem.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_op (torch.Tensor): regularization parameter for operator term in loss.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.

        Returns:
            loss (torch.Tensor): loss.
            loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
        """

        if bval is None:
            return sum(operator)

        loss_oper, op = self._loss_op(operator, lambda_op)

        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        lambda_op_normalized = lambda_prepare(operator, 1)
        lambda_bound_normalized = lambda_prepare(bval, 1)

        with torch.no_grad():
            loss_normalized = op @ lambda_op_normalized.T + \
                              bval_diff @ lambda_bound_normalized.T

        loss_dict = {
            "loss": loss,
            "loss_normalized": loss_normalized.detach(),
            "loss_oper": loss_oper.detach(),
            "loss_bnd": loss_bnd.detach(),
            "operator": operator.detach(),
            "bval_diff": bval_diff.detach()
        }

        return loss_dict


def get_PINN(layer_sizes, device):
    layers = []
    for i, j in zip(layer_sizes[:-1], layer_sizes[1:]):
        layer = torch.nn.Linear(i, j)
        layers.append(layer)
        layers.append(torch.nn.Tanh())
    layers = layers[:-1]
    return torch.nn.Sequential(*layers).to(device)
