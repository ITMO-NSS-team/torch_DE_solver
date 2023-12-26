import numpy as np
from tedeous.callbacks.callback import Callback


class EarlyStopping(Callback):
    def __init__(self, eps=0.01,
                 loss_window=1,
                 no_improvement_patience=1000,
                 patience=5,
                 abs_loss=0.01,
                 normalized_loss=False,
                 ):
        super().__init__()
        self.eps = eps
        self.loss_window = loss_window
        self.no_improvement_patience = no_improvement_patience
        self.patience = patience
        self.abs_loss = abs_loss
        self.normalized_loss = normalized_loss
        self.t = self.model.t
        self.mode = self.model.mode
        self._stop_dings = 0
        self.cur_loss = np.inf

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
                    self.model.apply(self.model._r)
                self._check = 'window_check'

    def _patience_check(self):
        """ Stopping criteria. We control the minimum loss and count steps
        when the current loss is bigger then min_loss. If these steps equal to
        no_improvement_patience parameter, the stopping criteria will be achieved.

        Args:
            no_improvement_patience (int): no improvement steps param.
        """
        if (self.t - self._t_imp_start) == self.no_improvement_patience and self._check is None:
            self._t_imp_start = self.t
            self._stop_dings += 1
            if self.mode in ('NN', 'autograd'):
                self.model.apply(self.model._r)
            self._check = 'patience_check'

    def _absloss_check(self):
        """ Stopping criteria. If current loss absolute value is lower then *abs_loss* param,
        the stopping criteria will be achieved.
        """
        if self.abs_loss is not None and self.cur_loss < self.abs_loss and self._check is None:
            self._stop_dings += 1
            self._check = 'absloss_check'

    def on_epoch_begin(self, logs=None):
        self._window_check()
        self._patience_check()
        self._absloss_check()

    def on_epoch_end(self, logs=None):
        if self._stop_dings < self.patience:
            self.model.stop_training = True
