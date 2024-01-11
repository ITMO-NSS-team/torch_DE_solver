import torch
from typing import Union, List, Any

from tedeous.data import Domain, Conditions, Equation
from tedeous.input_preprocessing import Operator_bcond_preproc
from tedeous.callbacks.callback_list import CallbackList
from tedeous.solution import Solution
from tedeous.optimizers.optimizer import Optimizer
from tedeous.utils import CacheUtils
from tedeous.optimizers.closure import Closure


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
        self._check = None

    def compile(
            self,
            mode: str,
            lambda_operator: Union[List[float], float],
            lambda_bound: Union[List[float], float],
            normalized_loss_stop: bool = False,
            h: float = 0.001,
            inner_order: str = '1',
            boundary_order: str = '2',
            derivative_points: int = 2,
            weak_form: List[callable] = None,
            tol: float = 0):

        self.mode = mode
        self.lambda_bound = lambda_bound
        self.lambda_operator = lambda_operator
        self.normalized_loss_stop = normalized_loss_stop
        self.weak_form = weak_form

        grid = self.domain.build(mode=mode)
        dtype = grid.dtype
        self.net.to(dtype)
        variable_dict = self.domain.variable_dict
        operator = self.equation.equation_lst
        bconds = self.conditions.build(variable_dict)

        self.equation_cls = Operator_bcond_preproc(grid, operator, bconds, h=h, inner_order=inner_order,
                                                   boundary_order=boundary_order).set_strategy(mode)
        
        self.solution_cls = Solution(grid, self.equation_cls, self.net, mode, weak_form,
                                     lambda_operator, lambda_bound, tol, derivative_points)

    def _model_save(
        self,
        save_model: bool,
        model_name: str):
        """ Model saving.

        Args:
            cache_utils (CacheUtils): CacheUtils class object.
            save_always (bool): flag for model saving.
            scaler (Any): GradScaler for CUDA.
            name (str): model name.
        """
        if save_model:
            if self.mode == 'mat':
                CacheUtils().save_model_mat(model=self.net,
                                            grid=self.solution_cls.grid,
                                            name=model_name)
            else:
                CacheUtils().save_model(model=self.net, name=model_name)

    def train(self,
              optimizer: Optimizer,
              epochs: int,
              info_string_every: Union[int, None] = None,
              mixed_precision: bool = False,
              save_model: bool = False,
              model_name: Union[str, None] = None,
              callbacks=None):

        self.t = 1
        self.stop_training = False

        callbacks = CallbackList(callbacks=callbacks, model=self)

        callbacks.on_train_begin()

        self.optimizer = optimizer.optimizer_choice(self.mode, self.net)

        closure = Closure(mixed_precision, self).get_closure(optimizer.optimizer)

        self.min_loss = torch.min(closure())

        self.cur_loss = self.min_loss

        while self.t < epochs and self.stop_training == False:
            callbacks.on_epoch_begin()

            self.optimizer.zero_grad()
            
            self.optimizer.step(closure)

            callbacks.on_epoch_end()

            self.t += 1

            if self.t % info_string_every == 0 and info_string_every is not None:
                loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
                info = 'Step = {} loss = {:.6f}.'.format(self.t, loss)
                print(info)

        callbacks.on_train_end()

        self._model_save(save_model, model_name)


        