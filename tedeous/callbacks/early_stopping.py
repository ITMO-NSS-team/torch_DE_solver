import numpy as np
from typing import Union
import torch
import datetime
from tedeous.callbacks.callback import Callback
from tedeous.utils import create_random_fn


class EarlyStopping(Callback):
    def __init__(self,
                 eps: float = 1e-5,
                 loss_window: int = 100,
                 no_improvement_patience: int = 1000,
                 patience: int = 5,
                 abs_loss: Union[float, None] = None,
                 normalized_loss: bool = False,
                 randomize_parameter: float = 1e-5,
                 info_string_every: Union[int, None] = None,
                 verbose: bool = True
                 ):
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

    def _line_create(self):
        """ Approximating last_loss list (len(last_loss)=loss_oscillation_window) by the line.

        Args:
            loss_oscillation_window (int): length of last_loss list.
        """
        self._line = np.polyfit(range(self.loss_window), self.last_loss, 1)

    def _window_check(self):
        """ Stopping criteria. We devide angle coeff of the approximating
        line (line_create()) on current loss value and compare one with *eps*
        """
        if self.t % self.loss_window == 0 and self._check is None:
            self._line_create()
            if abs(self._line[0] / self.cur_loss) < self.eps and self.t > 0:
                self._stop_dings += 1
                if self.mode in ('NN', 'autograd'):
                    self.model.net.apply(self._r)
                self._check = 'window_check'

    def _patience_check(self):
        """ Stopping criteria. We control the minimum loss and count steps
        when the current loss is bigger then min_loss. If these steps equal to
        no_improvement_patience parameter, the stopping criteria will be achieved.

        Args:
            no_improvement_patience (int): no improvement steps param.
        """
        if (self.t - self._t_imp_start) == self.no_improvement_patience and self._check is None:
            self._stop_dings += 1
            if self.mode in ('NN', 'autograd'):
                self.model.net.apply(self._r)
            self._check = 'patience_check'

    def _absloss_check(self):
        """ Stopping criteria. If current loss absolute value is lower then *abs_loss* param,
        the stopping criteria will be achieved.
        """
        if self.abs_loss is not None and self.cur_loss < self.abs_loss and self._check is None:
            self._stop_dings += 1
            self._check = 'absloss_check'

    def verbose_print(self):
        """

        Args:
            no_improvement_patience (int): no improvement steps param. (see patience_check())
            print_every (Union[None, int]): print or save after *print_every* steps.
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

        if self._check is not None or self.t % self.info_string_every == 0:
            try:
                self._line
            except:
                self._line_create()
            loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
            info = 'Step = {} loss = {:.6f} normalized loss line= {:.6f}x+{:.6f}. There was {} stop dings already.'.format(
                    self.t, loss, self._line[0] / loss, self._line[1] / loss, self._stop_dings)
            print(info)

        self._check = None

    def on_epoch_end(self, logs=None):
        self._window_check()
        self._patience_check()
        self._absloss_check()

        if self.cur_loss < self.min_loss:
            self.min_loss = self.model.cur_loss.item()
            self._t_imp_start = self.t
        try:
            self.last_loss[(self.t - 1) % self.loss_window] = self.cur_loss
        except:
            self.last_loss = np.zeros(self.loss_window) + float(self.min_loss)

        if self.verbose:
            self.verbose_print()
        if self._stop_dings >= self.patience:
            self.model.stop_training = True

    def on_epoch_begin(self, logs=None):
        self.t = self.model.t
        self.mode = self.model.mode
        self._check = self.model._check
        self.cur_loss = self.model.cur_loss
        self.min_loss = self.model.min_loss
