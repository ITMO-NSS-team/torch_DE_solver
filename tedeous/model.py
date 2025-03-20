import torch
from typing import Union, List
import tempfile
import os
import datetime
import copy

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
              reuse_nncg_flag: bool = False,
              rl_opt_flag: bool = False,
              models_concat_flag: bool = False,
              n_save_models: int = None,
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
            reuse_nncg_flag (bool): reuse of NNCG optimizer in optimizers chain. Default to False.
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

        self.stop_training = False
        callbacks = CallbackList(callbacks=callbacks, model=self)
        callbacks.on_train_begin()

        self.net = self.solution_cls.model

        # Initialize min_loss
        self.min_loss, _ = self.solution_cls.evaluate()
        self.cur_loss = self.min_loss

        print('[{}] initial (min) loss is {}'.format(datetime.datetime.now(), self.min_loss.item()))

        def execute_training_phase(epochs, reuse_nncg_flag=False, n_save_models=1):
            if not(models_concat_flag and rl_opt_flag):
                self.saved_models = []

            while self.t < epochs and not self.stop_training:
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

                if rl_opt_flag and self.t % (epochs // n_save_models) == 0:
                    current_model = copy.deepcopy(self.net)
                    self.saved_models.append(current_model)

                callbacks.on_epoch_end()
                self.t += 1

                if info_string_every is not None and self.t % info_string_every == 0:
                    loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
                    print(f'[{datetime.datetime.now()}] Step = {self.t}, loss = {loss:.6f}.')
            else:
                loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
                print(f'[{datetime.datetime.now()}] Step = {self.t}, loss = {loss:.6f}.')

            if rl_opt_flag and self.t % (epochs // n_save_models) == 0:
                current_model = copy.deepcopy(self.net)
                self.saved_models.append(current_model)

            if rl_opt_flag:
                return loss, self.saved_models

        def compute_reward(prev_error, current_error, method="diff"):
            """
            Calculates the reward for the agent.

            Args:
                prev_error (float): Error in the previous step.
                current_error (float): Error at the current step.
                method (str): The method for calculating the reward (“diff” or “absolute”).

            Returns:
                float: The value of the reward.
            """
            if method == "diff":
                return prev_error - current_error
            elif method == "absolute":
                return -current_error
            else:
                raise ValueError("Invalid reward method. Use 'diff' or 'absolute'.")

        if isinstance(optimizer, list) and rl_opt_flag:
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
            action_dim = env.action_space.n

            rl_agent = DQNAgent(state_dim, action_dim)

            total_reward = 0

            # We will learn the model with first optimizer here
            # # Correct action
            # action = rl_agent.select_action(state)

            # Optimization of the RL algorithm is implemented in the file rl_algorithms
            optimizers = optimizer.copy()

            # Stub action
            action = optimizer[0]

            optimizer = Optimizer(action['name'], action['params'])
            self.optimizer = optimizer.optimizer_choice(self.mode, self.net)
            closure = Closure(mixed_precision, self, reuse_nncg_flag=reuse_nncg_flag).get_closure(optimizer.optimizer)
            self.t = 1

            print(f'\nRL agent training: step {1}.')
            print(f'Using optimizer: {action["name"]} for {action["epochs"]} epochs.')
            loss, solver_models = execute_training_phase(
                action["epochs"],
                reuse_nncg_flag=reuse_nncg_flag,
                n_save_models=n_save_models
            )
            env.solver_models = solver_models
            env.current_loss = loss
            print(f'Finished optimizer {action["name"]}.')

            # input weights (for generate state) and loss (for calculate reward) to step method
            # first getting current weights and current losses
            state, reward, done, _ = env.step()  # model.train()

            # rl_agent.push_memory(state, action, reward, next_state, reward)
            # rl_agent.optimize_model()

            # state = next_state
            total_reward += reward

            callbacks.callbacks[1].save_every = self.t
            env.render()

            num_episodes = len(optimizers)
            # num_episodes = action_dim

            # # Optimization of the RL algorithm is implemented in the file rl_algorithms
            # optimizers = optimizer.copy()

            for i_episode in range(1, num_episodes):
                # if i_episode != 0:
                #     state = env.reset()
                #     state = check_device(state)
                # total_reward = 0

                # # Correct action
                # action = rl_agent.select_action(state)

                # Stub action
                action = optimizers[i_episode]

                optimizer = Optimizer(action['name'], action['params'])
                self.optimizer = optimizer.optimizer_choice(self.mode, self.net)
                closure = Closure(mixed_precision, self, reuse_nncg_flag=reuse_nncg_flag).get_closure(optimizer.optimizer)
                self.t = 1

                print(f'\nRL agent training: step {i_episode + 1}.')
                print(f'Using optimizer: {action["name"]} for {action["epochs"]} epochs.')
                loss, solver_models = execute_training_phase(
                    action["epochs"],
                    reuse_nncg_flag=reuse_nncg_flag,
                    n_save_models=n_save_models
                )
                env.solver_models = solver_models
                env.current_loss = loss
                print(f'Finished optimizer {action["name"]}.')

                # input weights (for generate state) and loss (for calculate reward) to step method
                # first getting current models and current losses
                next_state, reward, done, _ = env.step()

                rl_agent.push_memory(state, action, reward, next_state, reward)
                rl_agent.optimize_model()

                state = next_state
                total_reward += reward

                callbacks.callbacks[1].save_every = self.t
                env.render()

                if done or i_episode == len(optimizers):
                    print(f"Episode {i_episode}: Total Reward = {total_reward}")
                    break

        elif isinstance(optimizer, list) and not rl_opt_flag:
            optimizers_chain = optimizer.copy()
            for optimizer in optimizers_chain:
                opt_name = optimizer['name']
                opt_params = optimizer['params']
                opt_epochs = optimizer['epochs']
                optimizer = Optimizer(opt_name, opt_params)
                self.optimizer = optimizer.optimizer_choice(self.mode, self.net)

                closure = Closure(mixed_precision, self, reuse_nncg_flag=reuse_nncg_flag).get_closure(optimizer.optimizer)
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

                closure = Closure(mixed_precision, self, reuse_nncg_flag=reuse_nncg_flag).get_closure(optimizer.optimizer)
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
