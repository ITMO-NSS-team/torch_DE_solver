import torch
import numpy as np
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
from tedeous.utils import save_model_nn, save_model_mat
from tedeous.optimizers.closure import Closure
from tedeous.device import device_type

from tedeous.rl_algorithms import DQNAgent
from tedeous.rl_environment import EnvRLOptimizer


def raw_action_postproc(tup):
    optims_ar = ['Adam', 'RAdam', 'Adam', 'LBFGS', 'PSO', 'CSO', 'RMSprop']
    loss_ar = [0.1, 0.01, 0.001, 0.0001]
    epochs_ar = [10, 100, 1000]
    i_optim, i_loss, i_epochs = tup
    return {'name': optims_ar[i_optim], 'params': {'lr': loss_ar[i_loss]}, 'epochs': epochs_ar[i_epochs]}


def get_state_shape(loss_surface_params):
    min_x, max_x, xnum = loss_surface_params["x_range"]
    min_y, max_y = min_x, max_x
    step_size = (max_x - min_x) / xnum

    x_coords = torch.arange(min_x, max_x + step_size, step_size)
    y_coords = torch.arange(min_y, max_y + step_size, step_size)

    return tuple(torch.meshgrid(x_coords, y_coords)[0].shape)


class Model():
    """class for preprocessing"""

    def __init__(
            self,
            net: Union[torch.nn.Module, torch.Tensor],
            domain: Domain,
            equation: Equation,
            conditions: Conditions,
            batch_size: int = None):
        """
        Args:
            net (Union[torch.nn.Module, torch.Tensor]): neural network or torch.Tensor for mode *mat*
            grid (Domain): object of class Domain
            equation (Equation): object of class Equation
            conditions (Conditions): object of class Conditions
            batch_size (int): size of batch
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
        """ Compile model for training process.

        Args:
            mode (str): *mat, NN, autograd*
            lambda_operator (Union[List[float], float]): weight for operator term.
            It can be float for single equation or list of float for system.
            lambda_bound (Union[List[float], float]): weight for boundary term.
            It can be float for all types of boundary cond-ns or list of float for every condition type.
            normalized_loss_stop (bool, optional): loss with lambdas=1. Defaults to False.
            h (float, optional): increment for finite-difference scheme only for *NN*. Defaults to 0.001.
            inner_order (str, optional): order of finite-difference scheme *'1', '2'* for inner points.
            Only for *NN*. Defaults to '1'.
            boundary_order (str, optional): order of finite-difference scheme *'1', '2'* for boundary points.
            Only for *NN*. Defaults to '2'.
            derivative_points (int, optional): number of points for finite-difference scheme in *mat* mode.
            if derivative_points=2 the central scheme are used. Defaults to 2.
            weak_form (List[callable], optional): basis function for weak loss. Defaults to None.
            tol (float, optional): tolerance for causual loss. Defaults to 0.
            removed_domains (list): domains to be removed from the grid. Defaults to None.
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
        """ Model saving.

        Args:
            save_model (bool): save model or not.
            model_name (str): model name.
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
              optimizer: Union[Optimizer, list, dict],
              epochs: int,
              info_string_every: Union[int, None] = None,
              mixed_precision: bool = False,
              save_model: bool = False,
              model_name: Union[str, None] = None,
              callbacks: Union[List, None] = None,
              rl_opt_flag: bool = False,
              models_concat_flag: bool = False,
              n_save_models: int = None,
              equation_params: list = None,
              AE_model_params: dict = None,
              AE_train_params: dict = None,
              loss_surface_params: dict = None,
              n_trajectories: int = 100):
        """ train model.

        Args:
            optimizer (Union[Optimizer, list, dict]): the object of Optimizer class or dict, or list
            epochs (int): number of epoch for training.
            info_string_every (Union[int, None], optional): print loss state after *info_string_every* epoch. Defaults to None.
            mixed_precision (bool, optional): apply mixed precision for calculation. Defaults to False.
            save_model (bool, optional): save resulting model in cache. Defaults to False.
            model_name (Union[str, None], optional): model name. Defaults to None.
            callbacks (Union[List, None], optional): callbacks for training process. Defaults to None.
            rl_opt_flag (bool): use RL optimizer instead default. Defaults to False.
            n_save_models (int): number of points on the loss trajectory. Default to None.
            models_concat_flag (bool): concatenate loss tensors of models (for loss landscape create) or not. Default to False.
            equation_params (list): parameters (grid, domain, equation, boundaries) of experiment. Defaults to None.
            AE_model_params (dict): parameters of autoencoder model. Default to None.
            AE_train_params (dict): parameters of autoencoder train process. Default to None.
            loss_surface_params (dict): parameters of loss surface create. Default to None.
        """

        self.t = 1
        self.saved_models = []
        self.prev_to_current_optimizer_models = []

        self.stop_training = False
        callbacks = CallbackList(callbacks=callbacks, model=self)
        callbacks.on_train_begin()

        self.net = self.solution_cls.model

        # Initialize min_loss
        self.min_loss, _ = self.solution_cls.evaluate()
        self.cur_loss = self.min_loss

        print('[{}] initial (min) loss is {}'.format(datetime.datetime.now(), self.min_loss.item()))

        def execute_training_phase(epochs, reuse_nncg_flag=False, n_save_models=1):
            if not (models_concat_flag and rl_opt_flag):
                self.saved_models = []

            while self.t <= epochs and not self.stop_training:
                callbacks.on_epoch_begin()
                self.optimizer.zero_grad()

                # this fellow should be in NNCG closure, but since it calls closure many times,
                # it updates several time, which casuses instability

                if optimizer.optimizer == 'NNCG' and \
                        ((self.t - 1) % optimizer.params['precond_update_frequency'] == 0) and not reuse_nncg_flag:
                    grads = self.optimizer.gradient(self.cur_loss)
                    grads = torch.where(grads != grads, torch.zeros_like(grads), grads)
                    print('here t={} and freq={}'.format(self.t - 1, optimizer.params['precond_update_frequency']))
                    self.optimizer.update_preconditioner(grads)

                iter_count = 1 if self.batch_size is None else self.solution_cls.operator.n_batches
                for _ in range(iter_count):  # if batch mod then iter until end of batches else only once
                    if device_type() == 'cuda' and mixed_precision:
                        closure()
                    else:
                        self.optimizer.step(closure)
                    if optimizer.gamma is not None and self.t % optimizer.decay_every == 0:
                        optimizer.sheduler.step()

                # ERROR: when epochs < n_save_models!!!
                if rl_opt_flag:
                    current_model = copy.deepcopy(self.net)
                    self.saved_models.append(current_model)
                    self.prev_to_current_optimizer_models.append(current_model)

                callbacks.on_epoch_end()
                self.t += 1

                if info_string_every is not None and self.t % info_string_every == 0:
                    loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
                    print(f'[{datetime.datetime.now()}] Step = {self.t}, loss = {loss:.6f}.')
            else:
                loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
                print('while loop is over:', self.t, self.stop_training)
                print(f'[{datetime.datetime.now()}] Step = {self.t}, loss = {loss:.6f}.')

            if rl_opt_flag:
                current_model = copy.deepcopy(self.net)
                self.saved_models.append(current_model)
                self.prev_to_current_optimizer_models.append(current_model)
                indices_prev_to_current_models = np.linspace(0, len(self.prev_to_current_optimizer_models) - 1, 10, dtype=int)
                self.prev_to_current_optimizer_models = [self.prev_to_current_optimizer_models[i] for i in indices_prev_to_current_models]
                    
                if len(self.saved_models) >= n_save_models:
                    indices_saved_models = np.linspace(0, len(self.saved_models) - 1, 10, dtype=int)
                    self.saved_models = [self.saved_models[i] for i in indices_saved_models]
                else:
                    print("Using prev optimizer models")
                    self.saved_models = self.prev_to_current_optimizer_models

                return loss, self.saved_models

        if rl_opt_flag:
            env = EnvRLOptimizer(optimizer,
                                 equation_params=equation_params,
                                 callbacks=callbacks,
                                 AE_model_params=AE_model_params,
                                 AE_train_params=AE_train_params,
                                 loss_surface_params=loss_surface_params,
                                 n_save_models=n_save_models)

            # These objects must be created after the first optimizer is started
            state_dim = env.observation_space
            # state_dim = np.prod(env.observation_space.shape)
            action_dim = env.action_space

            memory_size = 1024  # ????

            rl_agent = DQNAgent(state_dim, action_dim, memory_size=memory_size, device=device_type())

            # # Optimization of the RL algorithm is implemented in the file rl_algorithms
            # optimizers = optimizer.copy()

            # n_dims = (1, 26, 26)  # CHANGE!!!
            # n_dims = (26, 26)

            state_shape = get_state_shape(loss_surface_params)

            # # state = torch init -> AE_model
            # state = torch.zeros(state_shape)
            # total_reward = 0
            # optimizers_history = []

            for traj in range(n_trajectories):
                print('\n############################################################################' +
                      f'\nStarting trajectory {traj + 1}/{n_trajectories} with a new initial point.')

                self.net = self.solution_cls.model
                self.t = 1

                # state = torch init -> AE_model
                total_reward = 0
                optimizers_history = []
                state = torch.zeros(state_shape)

                for i in itertools.count():
                    action_raw = rl_agent.select_action(state)
                    action = raw_action_postproc(action_raw)

                    # # Stub action
                    # action = optimizers[i]

                    reuse_nncg_flag = action["name"] == 'NNCG' if i > 0 else False

                    optimizer = Optimizer(action['name'], action['params'])
                    self.optimizer = optimizer.optimizer_choice(self.mode, self.net)
                    closure = Closure(mixed_precision, self, reuse_nncg_flag=reuse_nncg_flag).get_closure(
                        optimizer.optimizer)
                    self.t = 1

                    print('\n===========================================================================\n' +
                          f'\nRL agent training: step {i + 1}.'
                          f'\nTime: {datetime.datetime.now()}.'
                          f'\nUsing optimizer: {action["name"]} for {action["epochs"]} epochs.'
                          f'\nTotal Reward = {total_reward}.\n')

                    loss, solver_models = execute_training_phase(
                        action["epochs"],
                        reuse_nncg_flag=reuse_nncg_flag,
                        n_save_models=n_save_models
                    )
                    env.solver_models = solver_models
                    env.current_loss = 1 / loss

                    optimizers_history.append(action["name"])
                    print(f'\nPassed optimizer {action["name"]}.')

                    # input weights (for generate state) and loss (for calculate reward) to step method
                    # first getting current models and current losses
                    next_state, reward, done, _ = env.step()

                    if i != 0:
                        rl_agent.push_memory((state, next_state, action_raw, reward))

                    if rl_agent.replay_buffer.__len__() == memory_size:
                        rl_agent.optim_()

                    state = next_state
                    total_reward += reward

                    print(f'\nCurrent reward after {action["name"]} optimizer: {reward}.\n'
                          f'Total reward after using {", ".join(optimizers_history)} '
                          f'{"optimizers" if len(optimizers_history) > 1 else "optimizer"}: {total_reward}.\n')

                    callbacks.callbacks[1].save_every = self.t
                    env.render()

                    if done:
                        break

        elif isinstance(optimizer, list) and not rl_opt_flag:
            optimizers_chain = optimizer.copy()
            for optimizer in optimizers_chain:
                opt_name = optimizer['name']
                opt_params = optimizer['params']
                opt_epochs = optimizer['epochs']
                optimizer = Optimizer(opt_name, opt_params)
                self.optimizer = optimizer.optimizer_choice(self.mode, self.net)

                reuse_nncg_flag = opt_name == 'NNCG'
                closure = Closure(mixed_precision, self, reuse_nncg_flag=reuse_nncg_flag).get_closure(
                    optimizer.optimizer)
                self.t = 1

                print(f'\n[{datetime.datetime.now()}] Using optimizer: {opt_name} for {opt_epochs} epochs.')
                execute_training_phase(opt_epochs, reuse_nncg_flag=reuse_nncg_flag)
                print(f'[{datetime.datetime.now()}] Finished optimizer {opt_name}.')

        elif isinstance(optimizer, dict):
            optimizers_chain = optimizer.copy()
            for opt_name, opt_params in optimizers_chain.items():
                opt_param = opt_params[0]
                opt_epochs = opt_params[1]
                optimizer = Optimizer(opt_name, opt_param)
                self.optimizer = optimizer.optimizer_choice(self.mode, self.net)

                reuse_nncg_flag = opt_name == 'NNCG'
                closure = Closure(mixed_precision, self, reuse_nncg_flag=reuse_nncg_flag).get_closure(
                    optimizer.optimizer)
                self.t = 1

                print(f'\n[{datetime.datetime.now()}] Using optimizer: {opt_name} for {opt_epochs} epochs.')
                execute_training_phase(opt_epochs, reuse_nncg_flag=reuse_nncg_flag)
                print(f'[{datetime.datetime.now()}] Finished optimizer {opt_name}.')

        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer.optimizer_choice(self.mode, self.net)
            closure = Closure(mixed_precision, self).get_closure(optimizer.optimizer)
            execute_training_phase(epochs)

        callbacks.on_train_end()

        self._model_save(save_model, model_name)
