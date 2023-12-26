import torch
from typing import Union, List

from tedeous.data import Domain, Conditions, Equation
from tedeous.input_preprocessing import Operator_bcond_preproc
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

    def compile(
            self,
            mode: str = 'autograd',
            loss: str = 'default',
            h: float = None,
            inner_order: str = '1',
            boundary_order: str = '2',
            derivative_points: int = None):

        grid = self.domain.build(mode=mode)
        variable_dict = self.domain.variable_dict
        operator = self.equation.equation_lst
        bconds = self.conditions.build(variable_dict)

        equation_cls = Operator_bcond_preproc(
            grid,
            operator,
            bconds,
            h=h,
            inner_order=inner_order,
            boundary_order=boundary_order).set_strategy(mode)
        
        return equation_cls

    def train(self):
        pass
