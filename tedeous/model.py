import torch
from typing import Union, List

from tedeous.data import Domain, Conditions, Equation
from tedeous.input_preprocessing import Operator_bcond_preproc
from tedeous.callbacks.callback_list import CallbackList
from tedeous.solution import Solution


class Model():
    """class for preprocessing"""
    def __init__(
            self,
            net: Union[torch.nn.Module, torch.Tensor],
            domain: Domain,
            equation: Equation,
            conditions: Conditions):
        """
        Args:
            net (Union[torch.nn.Module, torch.Tensor]): neural network or torch.Tensor for mode *mat*
            grid (Domain): object of class Domain
            equation (Equation): object of class Equation
            conditions (Conditions): object of class Conditions
        """
        self.net = net
        self.domain = domain
        self.equation = equation
        self.conditions = conditions
        self.equation_cls = None
        self.mode = None

    def compile(
            self,
            mode: str,
            lambda_operator: Union[List[float], float],
            lambda_bound: Union[List[float], float],
            h: float = 0.001,
            inner_order: str = '1',
            boundary_order: str = '2',
            derivative_points: int = 2,
            weak_form: List[callable] = None,
            tol: float = 0):

        self.mode = mode
        self.lambda_bound = lambda_bound
        self.lambda_operator = lambda_operator
        self.weak_form = weak_form

        grid = self.domain.build(mode=mode)
        variable_dict = self.domain.variable_dict
        operator = self.equation.equation_lst
        bconds = self.conditions.build(variable_dict)

        self.equation_cls = Operator_bcond_preproc(grid, operator, bconds, h=h, inner_order=inner_order,
                                                   boundary_order=boundary_order).set_strategy(mode)
        
        self.solution_cls = Solution(grid, self.equation_cls, self.net, mode, weak_form,
                                     lambda_operator, lambda_bound, tol, derivative_points)

    def train(self,
              optimizer,
              epochs,
              verbose: bool = True,
              mixed_precision: bool = False,
              save_model: bool = False,
              callbacks=None):

        if callbacks is not None:
            callbacks = CallbackList(callbacks=callbacks)
        callbacks.set_model(self)

        callbacks.on_train_begin()

        epoch = 0
            