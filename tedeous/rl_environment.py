import torch
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from tedeous.callbacks.plot import Plots


def load_loss_surface():
    """Загрузка данных о поверхности ошибки."""
    return torch.load("loss_surface_data.pt")


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


class OptimizerEnv(gym.Env):
    def __init__(self, optimizer_configs: List[Dict]):
        super(OptimizerEnv, self).__init__()

        self.optimizer_configs = optimizer_configs
        self.current_optimizer = None
        self.current_error = None
        self.loss_surface = None
        self.error_history = []
        self.tolerance = 1e-4

        # Action - selecting an optimizer with its parameters
        self.action_space = spaces.Discrete(len(self.optimizer_configs))

        # State - error surface (can be an array)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(100, 100), dtype=np.float32)

    def reset(self):
        """Reset environment - load error surface, reset history to zero, select starting point."""

        raw_state = load_loss_surface()
        self.loss_surface = raw_state['grid_losses']
        self.current_error = np.max(self.loss_surface)  # Начинаем с максимальной ошибки
        self.error_history = [self.current_error]  # Запоминаем стартовое значение ошибки
        return self.loss_surface

    def step(self, action: int):
        """Applying an action (optimizer selection) and updating the state."""

        optimizer_params = self.optimizer_configs[action]
        self.current_optimizer = optimizer_params

        error_prev = self.current_error
        self.current_error = self.train_model(optimizer_params)  # Обучение модели

        # reward = compute_reward(error_prev, self.current_error)
        # self.error_history.append(self.current_error)
        #
        # done = self.current_error < self.tolerance

        return self.loss_surface, {}

    def render(self):
        """Display the current error and convergence history."""

        print(f"Optimizer: {self.current_optimizer}, Error: {self.current_error}")

        plt.figure(figsize=(10, 5))
        plt.plot(self.error_history, label='Error')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Error Dynamics")
        plt.legend()
        plt.show()

    def close(self):
        plt.close('all')
