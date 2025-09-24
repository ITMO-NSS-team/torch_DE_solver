import torch
import numpy as np
import torch.nn.init as init
import torch.nn as nn
from typing import Union, List
import tempfile
import os
import datetime
import copy
import itertools

from tedeous.data import Domain, Conditions, Equation
from tedeous.input_preprocessing import Operator_bcond_preproc
from tedeous.callbacks.callback_list import CallbackList
from tedeous.solution import Solution
from tedeous.optimizers.optimizer import Optimizer
from tedeous.utils import save_model_nn, save_model_mat, exact_solution_data
from tedeous.optimizers.closure import Closure
from tedeous.device import device_type


class Model():
    """
    class for preprocessing
    """


    def __init__(
            self,
            net: Union[torch.nn.Module, torch.Tensor],
            domain: Domain,
            equation: Equation,
            conditions: Conditions,
            batch_size: int = None
    ):
        """
        Initializes the Model class, preparing it for solving differential equations using a neural network. This involves setting up the network, problem domain, equation, and boundary conditions. The initialization also configures a temporary directory for caching intermediate results during the solution process.
        
                Args:
                    net (Union[torch.nn.Module, torch.Tensor]): The neural network to be trained, or a torch.Tensor for matrix-based approaches.
                    domain (Domain): The spatial or temporal domain over which the differential equation is defined.
                    equation (Equation): The differential equation to be solved.
                    conditions (Conditions): The boundary and initial conditions associated with the differential equation.
                    batch_size (int, optional): The size of the batch used during training. Defaults to None.
        
                Returns:
                    None
                
                Why:
                This initialization sets up all necessary components for solving the differential equation with a neural network, including the network architecture, problem definition, and training configurations.
        """
        self.net = net
        self.domain = domain
        self.equation = equation
        self.conditions = conditions

        self._check = None
        temp_dir = tempfile.gettempdir()
        folder_path = os.path.join(temp_dir, 'tedeous_cache/')
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            pass
        else:
            os.makedirs(folder_path)
        self._save_dir = folder_path
        self.batch_size = batch_size

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
            tol: float = 0,
            removed_domains: list = None):
        """
        Compiles the model by preparing the computational domain, defining the equation and boundary conditions, and initializing the appropriate solver. This process sets up the model for the training loop, enabling the approximation of differential equation solutions using neural networks.
        
                Args:
                    mode (str): Specifies the computational mode (*mat*, *NN*, or *autograd*) to determine the method for solving the differential equation.
                    lambda_operator (Union[List[float], float]): Weight(s) for the operator term in the loss function, controlling the regularization of the equation. Can be a single float for a single equation or a list of floats for a system of equations.
                    lambda_bound (Union[List[float], float]): Weight(s) for the boundary term in the loss function, controlling the enforcement of boundary conditions. Can be a single float for all boundary condition types or a list of floats for each condition type.
                    normalized_loss_stop (bool, optional): If True, the loss is normalized with lambdas set to 1. Defaults to False.
                    h (float, optional): Increment for the finite-difference scheme, used only in *NN* mode. Defaults to 0.001.
                    inner_order (str, optional): Order of the finite-difference scheme (*'1'* or *'2'*) for inner points, used only in *NN* mode. Defaults to '1'.
                    boundary_order (str, optional): Order of the finite-difference scheme (*'1'* or *'2'*) for boundary points, used only in *NN* mode. Defaults to '2'.
                    derivative_points (int, optional): Number of points for the finite-difference scheme in *mat* mode. If set to 2, the central scheme is used. Defaults to 2.
                    weak_form (List[callable], optional): List of basis functions for the weak formulation of the loss. Defaults to None.
                    tol (float, optional): Tolerance for the penalty in the *casual loss*. Defaults to 0.
                    removed_domains (list): List of domains to be removed from the grid. Defaults to None.
        
                Returns:
                    None
        """
        self.mode = mode
        self.lambda_bound = lambda_bound
        self.lambda_operator = lambda_operator
        self.normalized_loss_stop = normalized_loss_stop
        self.weak_form = weak_form
        self.removed_domains = removed_domains

        grid = self.domain.build(mode=mode, removed_domains=removed_domains)
        dtype = grid.dtype

        self.net.to(dtype)
        variable_dict = self.domain.variable_dict
        operator = self.equation.equation_lst
        bconds = self.conditions.build(variable_dict)

        self.equation_cls = Operator_bcond_preproc(grid, operator, bconds, h=h, inner_order=inner_order,
                                                   boundary_order=boundary_order).set_strategy(mode)

        if self.batch_size != None:
            if len(grid) < self.batch_size:
                self.batch_size = None

        self.solution_cls = Solution(grid, self.equation_cls, self.net, mode, weak_form,
                                     lambda_operator, lambda_bound, tol, derivative_points,
                                     batch_size=self.batch_size)

    def _model_save(
            self,
            save_model: bool,
            model_name: str):
        """
        Saves the trained neural network model.
        
        This method persists the learned parameters of the neural network,
        allowing for later use without retraining. The saving mechanism
        depends on the configured mode (e.g., 'mat' for MATLAB-compatible format
        or a PyTorch-native format). This ensures that the trained solution
        to the differential equation can be easily reused or deployed.
        
        Args:
            save_model (bool): A flag indicating whether to save the model.
            model_name (str): The desired name for the saved model file.
        
        Returns:
            None
        """
        if save_model:
            if self.mode == 'mat':
                save_model_mat(self._save_dir,
                               model=self.net,
                               domain=self.domain,
                               name=model_name)
            else:
                save_model_nn(self._save_dir, model=self.net, name=model_name)

    def train(self,
              optimizer: Optimizer,
              epochs: int,
              info_string_every: Union[int, None] = None,
              mixed_precision: bool = False,
              save_model: bool = False,
              model_name: Union[str, None] = None,
              callbacks: Union[List, None] = None):
        """
        Trains the neural network to approximate the solution of a differential equation.
        
                The training process involves iteratively refining the network's parameters using the provided optimizer and closure function.
                Callbacks are used to monitor and control the training process, allowing for actions such as early stopping or logging.
                The method evaluates the loss and updates the model's weights to minimize the difference between the predicted and actual solutions.
        
                Args:
                    optimizer (Optimizer): The optimizer instance used to update the network's weights.
                    epochs (int): The number of training epochs to perform.
                    info_string_every (Union[int, None], optional):  (Deprecated) Print loss state after *info_string_every* epoch. Defaults to None.
                    mixed_precision (bool, optional): Whether to use mixed precision training for faster computation. Defaults to False.
                    save_model (bool, optional): Whether to save the trained model to the cache. Defaults to False.
                    model_name (Union[str, None], optional): The name to use when saving the model. Defaults to None.
                    callbacks (Union[List, None], optional): A list of callbacks to execute during training. Defaults to None.
        
                Returns:
                    None
        """

        self.t = 1
        self.stop_training = False

        callbacks = CallbackList(callbacks=callbacks, model=self)

        callbacks.on_train_begin()

        self.net = self.solution_cls.model

        self.optimizer = optimizer.optimizer_choice(self.mode, self.net)

        closure = Closure(mixed_precision, self).get_closure(optimizer.optimizer)

        self.min_loss, _ = self.solution_cls.evaluate()

        self.cur_loss = self.min_loss

        print('[{}] initial (min) loss is {}'.format(
            datetime.datetime.now(), self.min_loss.item()))

        while self.t < epochs and self.stop_training is False:
            callbacks.on_epoch_begin()
            self.optimizer.zero_grad()

            iter_count = 1 if self.batch_size is None else self.solution_cls.operator.n_batches
            for _ in range(iter_count):  # if batch mod then iter until end of batches else only once
                if device_type() == 'cuda' and mixed_precision:
                    closure()
                else:
                    self.optimizer.step(closure)
                if optimizer.gamma is not None and self.t % optimizer.decay_every == 0:
                    optimizer.scheduler.step()
            callbacks.on_epoch_end()

            self.t += 1
            if info_string_every is not None:
                if self.t % info_string_every == 0:
                    print('This parameter info_string_every in train is obstolete, please use callbacks')

        callbacks.on_train_end()

        self._model_save(save_model, model_name)
