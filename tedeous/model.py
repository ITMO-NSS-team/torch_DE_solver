import torch
from typing import Union, List, Any

from tedeous.data import Domain, Conditions, Equation
from tedeous.input_preprocessing import Operator_bcond_preproc
from tedeous.callbacks.callback_list import CallbackList
from tedeous.solution import Solution
from tedeous.optimizers.optimizer import Optimizer
from tedeous.callbacks.cache import CacheUtils


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

    def _model_save(
        self,
        cache_utils: CacheUtils,
        save_always: bool,
        scaler: Any,
        name: str):
        """ Model saving.

        Args:
            cache_utils (CacheUtils): CacheUtils class object.
            save_always (bool): flag for model saving.
            scaler (Any): GradScaler for CUDA.
            name (str): model name.
        """
        if save_always:
            if self.mode == 'mat':
                cache_utils.save_model_mat(model=self.model, grid=self.grid, name=name)
            else:
                scaler = scaler if scaler else None
                cache_utils.save_model(model=self.model, name=name)

    def train(self,
              optimizer: Optimizer,
              epochs,
              mixed_precision: bool = False,
              save_model: bool = False,
              model_name: Union[str, None] = None,
              callbacks=None):

        self.t = 0
        self.stop_training = False

        callbacks = CallbackList(callbacks=callbacks, model=self)

        callbacks.on_train_begin()

        optimizer = optimizer.optimizer_choice(self.mode, self.net)

        while self.t < epochs:
            callbacks.on_epoch_begin()

            optimizer.zero_grad()
            
            loss, _ = self.solution_cls.evaluate()

            loss.backward()
            optimizer.step()

            callbacks.on_epoch_end()
            self.t += 1
        

        callbacks.on_train_end()


        