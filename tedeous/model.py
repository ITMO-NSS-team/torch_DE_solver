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

from tedeous.rl_algorithms import DQNAgent
from tedeous.rl_environment import EnvRLOptimizer


def get_state_shape(loss_surface_params):
    min_x, max_x, xnum = loss_surface_params["x_range"]
    min_y, max_y = min_x, max_x
    step_size = (max_x - min_x) / xnum

    x_coords = torch.arange(min_x, max_x + step_size, step_size)
    y_coords = torch.arange(min_y, max_y + step_size, step_size)

    return tuple(torch.meshgrid(x_coords, y_coords)[0].shape)

def get_tup_actions(optimizers):
    tupe_actions = []
    for opt in range(len(optimizers['type'])):
        for epoch in range(len(optimizers['epochs'])):
            for lr in range(len(optimizers['params'])):
                tupe_actions.append((opt, epoch, lr))
    return tupe_actions

def make_legend(tupe_dqn_class, optimizers):
    with open('legend.txt', 'a') as the_file:
        for i, el in enumerate(tupe_dqn_class):
            opt, epoch, lr = el
            type_ = optimizers['type'][opt]
            epochs_ = optimizers['epochs'][epoch]
            params_ = optimizers['params'][lr]
            the_file.write(f'{i}: {type_}, {epochs_}, {params_}\n')


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

        self.grid = self.domain.build(mode=mode, removed_domains=removed_domains)
        dtype = self.grid.dtype
        self.net.to(dtype)
        variable_dict = self.domain.variable_dict
        operator = self.equation.equation_lst
        bconds = self.conditions.build(variable_dict)

        self.equation_cls = Operator_bcond_preproc(self.grid, operator, bconds, h=h, inner_order=inner_order,
                                                   boundary_order=boundary_order).set_strategy(mode)

        if self.batch_size != None:
            if len(self.grid) < self.batch_size:
                self.batch_size = None

        self.solution_cls = Solution(self.grid, self.equation_cls, self.net, mode, weak_form,
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

    def reinit_weights(self, m):
        # Если это линейный слой
        if isinstance(m, nn.Linear):
            # Например, инициализация Ксавьера
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def train(self,
              optimizer: Union[Optimizer, list, dict],
              epochs: int,
              info_string_every: Union[int, None] = None,
              mixed_precision: bool = False,
              save_model: bool = False,
              model_name: Union[str, None] = None,
              callbacks: Union[List, None] = None,
              rl_agent_params: dict = None,
              models_concat_flag: bool = False,
              equation_params: list = None,
              AE_model_params: dict = None,
              AE_train_params: dict = None,
              loss_surface_params: dict = None):
        """ train model.

        Args:
            optimizer (Union[Optimizer, list, dict]): the object of Optimizer class or dict, or list
            epochs (int): number of epoch for training.
            info_string_every (Union[int, None], optional): print loss state after *info_string_every* epoch. Defaults to None.
            mixed_precision (bool, optional): apply mixed precision for calculation. Defaults to False.
            save_model (bool, optional): save resulting model in cache. Defaults to False.
            model_name (Union[str, None], optional): model name. Defaults to None.
            callbacks (Union[List, None], optional): callbacks for training process. Defaults to None.
            rl_agent_params (dict): dictionary with rl agent parameters. Defaults to None.
            models_concat_flag (bool): concatenate loss tensors of models (for loss landscape create) or not. Defaults to False.
            equation_params (list): parameters (grid, domain, equation, boundaries) of experiment. Defaults to None.
            AE_model_params (dict): parameters of autoencoder model. Defaults to None.
            AE_train_params (dict): parameters of autoencoder train process. Defaults to None.
            loss_surface_params (dict): parameters of loss surface create. Defaults to None.
            exact_solution_func (callable): exact solution function for error calculate. Defaults to None.
        """

        self.t = 1
        self.saved_models = []
        if rl_agent_params:
            self.prev_to_current_optimizer_models = []
            self.rl_agent = None
            self.rl_penalty = 0

        self.stop_training = False
        callbacks = CallbackList(callbacks=callbacks, model=self)
        callbacks.on_train_begin()

        self.net = self.solution_cls.model

        # Initialize min_loss
        self.min_loss, _ = self.solution_cls.evaluate()
        self.cur_loss = self.min_loss

        print('[{}] initial (min) loss is {}'.format(datetime.datetime.now(), self.min_loss.item()))

        def execute_training_phase(epochs, new_graph_flag=None, n_save_models=1, stuck_threshold=50):
            if not (models_concat_flag and rl_agent_params):
                self.saved_models = []

            loss_history = []

            while self.t < epochs and not self.stop_training:
                callbacks.on_epoch_begin()
                self.optimizer.zero_grad()

                if rl_agent_params:
                    callbacks.callbacks[0]._stop_dings = 0

                # this fellow should be in NNCG closure, but since it calls closure many times,
                # it updates several time, which casuses instability

                if optimizer.optimizer == 'NNCG' and ((self.t - 1) % optimizer.params['precond_update_frequency'] == 0):
                    if new_graph_flag:
                        with torch.autocast(
                                device_type=device_type(),
                                dtype=self.grid.dtype,
                                enabled=mixed_precision
                        ):
                            loss, loss_normalized = self.solution_cls.evaluate()
                        grads = self.optimizer.gradient(loss)
                        new_graph_flag = None
                    else:
                        grads = self.optimizer.gradient(self.cur_loss)

                    grads = torch.where(grads != grads, torch.zeros_like(grads), grads)
                    print('here t={} and freq={}'.format(self.t - 1, optimizer.params['precond_update_frequency']))
                    self.optimizer.update_preconditioner(grads)

                iter_count = 1 if self.batch_size is None else self.solution_cls.operator.n_batches
                for _ in range(iter_count):  # if batch mod then iter until end of batches else only once
                    if device_type() == 'cuda' and mixed_precision:
                        closure()
                        # loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
                        # loss_history.append(loss)
                    else:
                        self.optimizer.step(closure)
                    if optimizer.gamma is not None and self.t % optimizer.decay_every == 0:
                        optimizer.sheduler.step()

                if rl_agent_params:
                    current_model = copy.deepcopy(self.net)
                    self.saved_models.append(current_model)
                    # self.prev_to_current_optimizer_models.append(current_model)

                loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
                loss_history.append(loss)

                callbacks.on_epoch_end()
                self.t += 1

                if info_string_every is not None and self.t % info_string_every == 0:
                    # loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
                    # loss_history.append(loss)
                    print(f'[{datetime.datetime.now()}] Step = {self.t}, loss = {loss:.6f}.')
            else:
                loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
                loss_history.append(loss)
                print(f'[{datetime.datetime.now()}] Step = {self.t}, loss = {loss:.6f}.')

            if rl_agent_params:
                current_model = copy.deepcopy(self.net)
                self.saved_models.append(current_model)

                loss_history = loss_history[-stuck_threshold:]

                if self.stop_training:
                    print(f"\nLocal min!!!")
                    self.rl_penalty = -1
                    callbacks.callbacks[0]._stop_dings = 0
                    self.stop_training = False

                if loss_history[-1] == np.nan:
                    self.rl_penalty = -1

                indices_saved_models = np.linspace(0, len(self.saved_models) - 1, n_save_models, dtype=int)
                self.saved_models = [self.saved_models[i] for i in indices_saved_models]

                return loss_history[-1], self.saved_models

        if rl_agent_params:
            env = EnvRLOptimizer(optimizer,
                                 equation_params=equation_params,
                                 callbacks=callbacks,
                                 AE_model_params=AE_model_params,
                                 AE_train_params=AE_train_params,
                                 loss_surface_params=loss_surface_params,
                                 n_save_models=rl_agent_params['n_save_models'],
                                 tolerance=rl_agent_params["tolerance"])

            # These objects must be created after the first optimizer is started
            n_observation = env.observation_space
            # state_dim = np.prod(env.observation_space.shape)
            n_action = env.action_space

            rl_agent = DQNAgent(n_observation,
                                n_action,
                                memory_size=rl_agent_params["rl_buffer_size"],
                                device=device_type(),
                                batch_size=rl_agent_params["rl_batch_size"])

            # Optimization of the RL algorithm is implemented in the file rl_algorithms
            optimizers = optimizer.copy()

            state_shape = get_state_shape(loss_surface_params)

            # # state = torch init -> AE_model
            # state = torch.zeros(state_shape)
            # total_reward = 0
            # optimizers_history = []

            done = None
            idx_traj = 0

            grid = self.domain.build('NN').to(device_type())
            variable_dict = self.domain.variable_dict
            bconds = self.conditions.build(variable_dict)

            # make_legend(tupe_dqn_class, optimizers)

            while rl_agent_params['n_trajectories'] - idx_traj > 0:
                self.net = self.solution_cls.model
                for m in self.net.modules():
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.xavier_normal_(m.weight)
                        torch.nn.init.zeros_(m.bias)
                self.t = 1

                # state = torch init -> AE_model
                total_reward = 0
                optimizers_history = []
                state = {"loss_total": torch.zeros(state_shape),
                         "loss_oper": torch.zeros(state_shape),
                         "loss_bnd": torch.zeros(state_shape)}

                # state = torch.tensor()

                print('\n############################################################################' +
                      f'\nStarting trajectory {idx_traj + 1}/{rl_agent_params["n_trajectories"]} ' +
                      'with a new initial point.')

                for i in itertools.count():
                    # state = torch.stack((state['loss_oper'], state['loss_bnd']), dim=0)
                    action, action_raw = rl_agent.select_action(state)
                    action_raw[2]['epochs'] = action_raw[1]
                    action_raw = (action_raw[0], action_raw[2])
                    # action_raw = tupe_dqn_class[dqn_class]
                    print(f"\naction = {action}")
                    # i_optim, i_epochs, i_loss = action_raw
                    # action = {
                    #     'type': optimizers['type'][i_optim],
                    #     'params': {'lr': optimizers['params'][i_loss]},
                    #     'epochs': optimizers['epochs'][i_epochs]
                    # }

                    optimizer = Optimizer(action['type'], action['params'])
                    self.optimizer = optimizer.optimizer_choice(self.mode, self.net)
                    closure = Closure(mixed_precision, self).get_closure(optimizer.optimizer)
                    self.t = 1

                    print('\n===========================================================================\n' +
                          f'\nRL agent training: step {i + 1}.'
                          f'\nTime: {datetime.datetime.now()}.'
                          f'\nUsing optimizer: {action["type"]} for {action["epochs"]} epochs.'
                          f'\nTotal Reward = {total_reward}.\n')

                    loss, solver_models = execute_training_phase(
                        action["epochs"],
                        new_graph_flag=i,
                        n_save_models=rl_agent_params['n_save_models'],
                        stuck_threshold=rl_agent_params['stuck_threshold'],
                        min_loss_change=rl_agent_params['min_loss_change'],
                        min_grad_norm=rl_agent_params['min_grad_norm']
                    )

                    if loss != loss:
                        self.rl_penalty = 0
                        break

                    env.rl_penalty = self.rl_penalty

                    if solver_models is None:
                        print("Solver models are None!!!")

                    if len(solver_models) < rl_agent_params['n_save_models']:
                        print(f"Current number of solver models: {len(solver_models)}. "
                              f"\nRight number = {rl_agent_params['n_save_models']}")

                    net = self.net.to(device_type())

                    if callable(rl_agent_params["exact_solution"]):
                        operator_rmse = torch.sqrt(
                            torch.mean((rl_agent_params["exact_solution"](grid).reshape(-1, 1) - net(grid)) ** 2)
                        )
                    else:
                        exact = exact_solution_data(grid, rl_agent_params["exact_solution"],
                                                    equation_params[-1][0], equation_params[-1][-1],
                                                    t_dim_flag='t' in list(self.domain.variable_dict.keys()))
                        net_predicted = net(grid)
                        operator_rmse = torch.sqrt(torch.mean((exact.reshape(-1, 1) - net_predicted) ** 2))

                    boundary_rmse = torch.sum(torch.tensor([
                        torch.sqrt(torch.mean((bconds[i]["bval"].reshape(-1, 1) - net(bconds[i]["bnd"])) ** 2))
                        for i in range(len(bconds))]))

                    env.solver_models = solver_models
                    env.reward_params = {
                        "operator": {
                            "error": operator_rmse,
                            "coeff": rl_agent_params["reward_operator_coeff"]
                        },
                        "bconds": {
                            "error": boundary_rmse,
                            "coeff": rl_agent_params["reward_boundary_coeff"]
                        }
                    }

                    optimizers_history.append(action["type"])
                    print(f'\nPassed optimizer {action["type"]}.')

                    # input weights (for generate state) and loss (for calculate reward) to step method
                    # first getting current models and current losses
                    next_state, reward, done, _ = env.step()

                    if done == 1:
                        reward += 100
                    elif done == 0:
                        reward -= 0.01 * i
                    elif done == -1:
                        reward = torch.tensor(-100, dtype=torch.int8)

                    # if i != 0:
                    #     rl_agent.push_memory((state, next_state, action_raw, reward))
                    # else:
                    #     rl_agent.steps_done -= 1
                    rl_agent.push_memory((state, next_state, action_raw, reward, abs(done)))
                    # for _ in range(32):
                    #     rl_agent.push_memory((state, next_state, dqn_class, reward))

                    if rl_agent.replay_buffer.__len__() % rl_agent_params["rl_batch_size"] == 0:
                        rl_agent.optim_()
                        # rl_agent.render_Q_function()

                    state = next_state
                    total_reward += reward

                    print(f'\nCurrent reward after {action["type"]} optimizer: {reward}.\n'
                          f'Total reward after using {", ".join(optimizers_history)} '
                          f'{"optimizers" if len(optimizers_history) > 1 else "optimizer"}: {total_reward}.\n'
                          f'\ndone = {done}')

                    callbacks.callbacks[1].save_every = self.t
                    env.render()

                    if done == 1:
                        break
                    elif done == 0:
                        continue
                    elif done == -1:
                        self.rl_penalty = 0
                        break

                if done == 1:
                    idx_traj += 1

            self.net = rl_agent.model

        if isinstance(optimizer, list):
            optimizers_chain = optimizer.copy()
            for i_opt in range(len(optimizers_chain)):
                opt_name = optimizers_chain[i_opt]['name']
                opt_params = optimizers_chain[i_opt]['params']
                opt_epochs = optimizers_chain[i_opt]['epochs']
                optimizer = Optimizer(opt_name, opt_params)

                self.optimizer = optimizer.optimizer_choice(self.mode, self.net)
                closure = Closure(mixed_precision, self).get_closure(optimizer.optimizer)
                self.t = 1

                print(f'\n[{datetime.datetime.now()}] Using optimizer: {opt_name} for {opt_epochs} epochs.')
                execute_training_phase(opt_epochs, new_graph_flag=i_opt)
                print(f'[{datetime.datetime.now()}] Finished optimizer {opt_name}.')

        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer.optimizer_choice(self.mode, self.net)
            closure = Closure(mixed_precision, self).get_closure(optimizer.optimizer)
            execute_training_phase(epochs)

        callbacks.on_train_end()

        self._model_save(save_model, model_name)