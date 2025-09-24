import numpy as np
from typing import Union
import torch
import datetime
from tedeous.callbacks.callback import Callback
from tedeous.utils import create_random_fn


class EarlyStopping(Callback):
    """
    Class for using adaptive stop criterias at training process.
    """

    def __init__(self,
                 eps: float = 1e-5,
                 loss_window: int = 100,
                 no_improvement_patience: int = 1000,
                 patience: int = 5,
                 abs_loss: Union[float, None] = None,
                 normalized_loss: bool = False,
                 randomize_parameter: float = 1e-5,
                 info_string_every: Union[int, None] = None,
                 verbose: bool = True,
                 save_best: bool = False
                 ):
        """
        Initializes the EarlyStopping criteria for monitoring the training process of a neural network-based differential equation solver.
        
                This class helps to stop the training process if the loss doesn't improve enough,
                avoiding overfitting and saving computational resources. It monitors the loss over a window
                and stops the training when the loss stagnates or increases.
        
                Args:
                    eps (float, optional): A small threshold to consider an improvement in loss. Defaults to 1e-5.
                    loss_window (int, optional): The number of recent losses to average for trend estimation. Defaults to 100.
                    no_improvement_patience (int, optional): How many steps to wait after the last improvement. Defaults to 1000.
                    patience (int, optional): How many times the "no improvement" condition must be met to stop training. Defaults to 5.
                    abs_loss (Union[float, None], optional): An absolute target loss value; stop if the loss goes below this. Defaults to None.
                    normalized_loss (bool, optional): Whether to use a normalized loss (all lambdas=1) for early stopping. Defaults to False.
                    randomize_parameter (float, optional): Magnitude of random noise added to model weights to escape local minima. Defaults to 1e-5.
                    info_string_every (Union[int, None], optional): Frequency (in steps) of printing loss information. Defaults to None.
                    verbose (bool, optional): Whether to print detailed information about the early stopping process. Defaults to True.
                    save_best (bool, optional): Whether to save the model weights corresponding to the lowest loss encountered. Defaults to False.
        """
        super().__init__()
        self.eps = eps
        self.loss_window = loss_window
        self.no_improvement_patience = no_improvement_patience
        self.patience = patience
        self.abs_loss = abs_loss
        self.normalized_loss = normalized_loss
        self._stop_dings = 0
        self._t_imp_start = 0
        self._r = create_random_fn(randomize_parameter)
        self.info_string_every = info_string_every if info_string_every is not None else np.inf
        self.verbose = verbose
        self.save_best=save_best
        self.best_model=None



    def _line_create(self):
        """
        Approximates the trend of recent loss values using linear regression.
        
        This method fits a line to the `last_loss` values, which represent the loss history
        within a defined window. This approximation helps in assessing whether the training
        loss is consistently decreasing, indicating convergence towards a solution of the
        differential equation.
        
        Args:
            None
        
        Returns:
            None
        """
        self._line = np.polyfit(range(self.loss_window), self.last_loss, 1)

    def _window_check(self):
        """
        Checks for early stopping based on the trend of the loss function over a window.
        
        This method assesses whether the training should be stopped early by analyzing the rate of change of the loss.
        It approximates the loss trend with a line and checks if the ratio of the line's slope to the current loss
        is below a specified threshold (*eps*). If the loss is not decreasing sufficiently, and the training has
        progressed beyond a minimum point, the method initiates the stopping procedure. This helps to prevent
        overfitting by halting training when the model's improvement plateaus.
        
        Args:
            None
        
        Returns:
            None
        """
        if self.t % self.loss_window == 0 and self._check is None:
            self._line_create()
            if abs(self._line[0] / self.model.cur_loss) < self.eps and self.t > 0:
                self._stop_dings += 1
                if self.mode in ('NN', 'autograd'):
                    self.model.net.apply(self._r)
                self._check = 'window_check'

    def _patience_check(self):
        """
        Checks if the training should be stopped based on the patience criterion.
        
        This method monitors the training loss and stops the training process if the loss
        has not improved for a specified number of epochs (patience). This prevents
        overfitting and ensures that the model converges to a reasonable solution
        within a practical timeframe when solving differential equations using neural networks.
        
        Args:
            None
        
        Returns:
            None
        """
        if (self.t - self._t_imp_start) == self.no_improvement_patience and self._check is None:
            self._stop_dings += 1
            self._t_imp_start = self.t
            if self.mode in ('NN', 'autograd'):
                if self.save_best:
                    self.model.net=self.best_model
                self.model.net.apply(self._r)
            self._check = 'patience_check'

    def _absloss_check(self):
        """
        Checks if the absolute value of the current loss is below a specified threshold, indicating a potential convergence in the neural network's solution to the differential equation. This is one of the criteria used to determine when to stop the training process, preventing overfitting and saving computational resources.
        
        Args:
            self (EarlyStopping): Instance of the EarlyStopping class.
        
        Returns:
            None. Modifies the internal state of the EarlyStopping object by incrementing the stop counter and setting the check flag if the condition is met.
        """
        if self.abs_loss is not None and self.model.cur_loss < self.abs_loss and self._check is None:
            self._stop_dings += 1
            self._check = 'absloss_check'

    def verbose_print(self):
        """
        Prints information about the current loss, step, and stopping criteria during training.
        
                This information helps monitor the training process and understand why the training might be stopping. It provides insights into whether the model is oscillating, has reached a plateau, or has achieved a satisfactory loss value, aiding in debugging and optimization.
        
                Args:
                    None
        
                Returns:
                    None
        """

        if self._check == 'window_check':
            print('[{}] Oscillation near the same loss'.format(
                            datetime.datetime.now()))
        elif self._check == 'patience_check':
            print('[{}] No improvement in {} steps'.format(
                        datetime.datetime.now(), self.no_improvement_patience))
        elif self._check == 'absloss_check':
            print('[{}] Absolute value of loss is lower than threshold'.format(
                                                        datetime.datetime.now()))

        if self._check is not None:
            try:
                self._line
            except:
                self._line_create()
            loss = self.model.cur_loss.item() if isinstance(self.model.cur_loss, torch.Tensor) else self.mdoel.cur_loss
            info = '[{}] Step = {} loss = {:.6f} normalized loss line= {:.6f}x+{:.6f}. There was {} stop dings already.'.format(
                    datetime.datetime.now(), self.t, loss, self._line[0] / loss, self._line[1] / loss, self._stop_dings)
            print(info)

    def info_print(self):
        """
        Prints a detailed information string during the training process. This information includes the current timestamp, training step, loss value, normalized loss line parameters, and the number of early stopping events encountered. This provides insight into the training progress and helps monitor the solver's performance.
        
                Args:
                    None
        
                Returns:
                    None
        """
        try:
            self._line
        except:
            self._line_create()
        loss = self.model.cur_loss.item() if isinstance(self.model.cur_loss, torch.Tensor) else self.mdoel.cur_loss
        info = '[{}] Step = {} loss = {:.6f} normalized loss line= {:.6f}x+{:.6f}. There was {} stop dings already.'.format(
                datetime.datetime.now(), self.t, loss, self._line[0] / loss, self._line[1] / loss, self._stop_dings)
        print(info)

    def on_epoch_end(self, logs=None):
        """
        Handles end-of-epoch actions, focusing on training dynamics and early stopping.
        
                This method evaluates stopping criteria based on loss trends, saves the model if a new best loss is achieved,
                and manages verbosity. It ensures the training process adapts to the loss landscape,
                preventing overfitting and optimizing for efficient convergence towards a solution of the differential equation.
        
                Args:
                    logs (dict, optional): Metric results from the current epoch. Defaults to None.
        
                Returns:
                    None
        """
        self._window_check()
        self._patience_check()
        self._absloss_check()

        if self.model.cur_loss < self.model.min_loss:
            self.model.min_loss = self.model.cur_loss
            if self.save_best:
                self.best_model=self.model.net
            self._t_imp_start = self.t

        if self.t % self.info_string_every == 0:
            self.info_print()

        if self.verbose:
            self.verbose_print()
        if self._stop_dings >= self.patience:
            self.model.stop_training = True
            if self.save_best:
                self.model.net=self.best_model
        self._check = None

    def on_epoch_begin(self, logs=None):
        """
        Updates the callback's internal state at the start of each epoch.
        
                This method retrieves the current training step, mode, and check flag from the model to track the training progress.
                It also stores the previous loss value in a circular buffer, which is initialized with the minimum loss value if not already initialized.
                This is done to keep track of recent loss values, which is essential for determining when the training should be stopped to prevent overfitting
                and ensure the model generalizes well to unseen data by monitoring the loss trend.
        
                Args:
                    logs: Optional dictionary of logs.
        
                Returns:
                    None
        """
        self.t = self.model.t
        self.mode = self.model.mode
        self._check = self.model._check
        try:
            self.last_loss[(self.t - 3) % self.loss_window] = self.model.cur_loss
        except:
            self.last_loss = np.zeros(self.loss_window) + float(self.model.min_loss)
