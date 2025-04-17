import gym
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Union

from landscape_visualization._aux.plot_loss_surface import PlotLossSurface
from landscape_visualization._aux.visualization_model import VisualizationModel
from landscape_visualization._aux.early_stopping_plot import EarlyStopping

from tedeous.optimizers.optimizer import Optimizer
from tedeous.callbacks.callback_list import CallbackList


def compute_reward(reward_params, prev_reward, method="diff"):
    """
    Calculates the reward for the agent.

    Args:
        reward_params (dict): dictionary with operator and boundary error value and coefficients.
        prev_reward (float): previous value of reward.
        method (str): The method for calculating the reward (“diff” or “absolute”).
    Returns:
        float: The value of the reward.
    """
    current_reward = reward_params["operator"]["coeff"] * reward_params["operator"]["error"] + \
        reward_params["bconds"]["coeff"] * reward_params["bconds"]["error"]

    if method == "diff":
        return prev_reward - current_reward
    elif method == "absolute":
        return -current_reward
    else:
        raise ValueError("Invalid reward method. Use 'diff' or 'absolute'.")


class EnvRLOptimizer(gym.Env):
    def __init__(self,
                 optimizers: dict,
                 equation_params: list = None,
                 loss_surface_params: dict = None,
                 AE_model_params: dict = None,
                 AE_train_params: dict = None,
                 reward_method: str = "absolute",
                 callbacks: Union[CallbackList, List, None] = None,
                 n_save_models: int = None,
                 tolerance: float = 1e-2):
        super(EnvRLOptimizer, self).__init__()

        self.optimizers = optimizers
        self.solver_models = None
        self.reward_params = None
        self.rl_penalty = 0
        self.raw_states_dict = {}

        self.AE_model_params = AE_model_params
        self.AE_train_params = AE_train_params
        self.loss_surface_params = loss_surface_params
        self.equation_params = equation_params
        self.reward_method = reward_method
        self.callbacks = callbacks

        self.visualization_model = VisualizationModel(**self.AE_model_params)
        self.plot_loss_surface = None

        # Размерность нужно вытягивать из кода loss landscape, она будет постоянной,
        # т.к. action_dim - список оптимизаторов, он не меняется
        # state_dim - размерность поверхности, мы используем латентное 2D пространство, для генерации поверхности

        # Action - selecting an optimizer with its parameters
        # self.action_space = spaces.Discrete(len(self.optimizer_configs))
        self.action_space = {key: len(value) for key, value in optimizers.items()}

        # # State - loss surface (can be an array)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.visualization_model.latent_dim,
        #                                     dtype=np.float32)
        # observation_space = 3
        self.observation_space = self.visualization_model.latent_dim + 1

        self.current_reward = None
        self.reward_history = []
        self.tolerance = tolerance
        self.counter = 1
        self.n_save_models = n_save_models

    def reset(self):
        """Reset environment - load error surface, reset history to zero, select starting point."""
        self.current_reward = self.reward_history[-1]
        self.counter += 1

    def step(self):
        """Applying an action (optimizer selection) and updating the state."""

        finetune_AE_model = self.AE_train_params['finetune_AE_model']
        batch_size = self.AE_train_params['batch_size']
        every_epoch = self.AE_train_params['every_epoch']
        learning_rate = self.AE_train_params['learning_rate']
        resume = self.AE_train_params['resume']
        AE_params = self.AE_train_params[
            'other_RL_epoch_AE_params' if finetune_AE_model else 'first_RL_epoch_AE_params'
        ]

        epochs = AE_params['epochs']
        patience_scheduler = AE_params['patience_scheduler']
        cosine_scheduler_patience = AE_params['cosine_scheduler_patience']

        optimizer = Optimizer('RMSprop', {'lr': learning_rate}, cosine_scheduler_patience=cosine_scheduler_patience)
        cb_es = EarlyStopping(patience=patience_scheduler)

        AEmodel = self.visualization_model.train(
            optimizer, epochs, every_epoch, batch_size, resume,
            callbacks=[cb_es], solver_models=self.solver_models, finetune_AE_model=finetune_AE_model
        )

        self.loss_surface_params['solver_models'] = self.solver_models
        self.loss_surface_params['AE_model'] = AEmodel

        self.plot_loss_surface = PlotLossSurface(**self.loss_surface_params)
        self.plot_loss_surface.counter = self.counter

        self.raw_states_dict = self.plot_loss_surface.save_equation_loss_surface(*self.equation_params)

        if len(self.reward_history) == 0:
            prev_reward = 0
        else:
            prev_reward = self.reward_history[-1]
            # min(self.loss_history[-10:]) if len(self.loss_history) > 9 else self.loss_history[-1]

        self.current_reward = compute_reward(
            self.reward_params, prev_reward, method=self.reward_method
        ) + self.rl_penalty

        self.reward_history.append(self.current_reward)

        done = (abs(self.current_reward) < self.tolerance) + self.rl_penalty

        return self.raw_states_dict, self.current_reward, done, {}

    def render(self):
        """Display the current error and convergence history."""

        self.reset()

        # print(f"Optimizer: {self.current_optimizer['name']}, Loss: {self.current_loss}")

        # Plotting PDE solution
        self.callbacks.on_epoch_end()
        self.callbacks.callbacks[1].save_every = 0.1

        # # Plotting loss landscape
        # if self.rl_penalty != -1:
        #     self.plot_loss_surface.plotting_equation_loss_surface(*self.equation_params)

    def close(self):
        plt.close('all')
