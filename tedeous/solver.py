"""Module for constructing computational pipline for P/O DE solutions."""

import os
import datetime
from typing import Union, List, Any
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch


from tedeous.device import check_device, device_type
from tedeous.solution import Solution
from tedeous.optimizers import PSO


def grid_format_prepare(
    coord_list: List,
    mode: str = 'NN') -> torch.Tensor:
    """ Prepares grid to a general form. Further, formatted grid can be processed
    by Points_type.point_typization for 'NN' and 'autograd' methods.

    Args:
        coord_list (List): list with coordinates.
        mode (str, optional): Calculation method. (i.e., "NN", "autograd", "mat").
        Defaults to 'NN'.

    Returns:
        torch.Tensor: grid in a general form.
    """

    device = device_type()
    if isinstance(coord_list, torch.Tensor):
        print('Grid is a tensor, assuming old format, no action performed')
        return check_device(coord_list)
    elif mode in ('NN', 'autograd'):
        if len(coord_list) == 1:
            coord_list = torch.tensor(coord_list).float().to(device)
            grid = coord_list.reshape(-1, 1)
        else:
            coord_list_tensor = []
            for item in coord_list:
                if isinstance(item, (np.ndarray)):
                    coord_list_tensor.append(torch.from_numpy(item).to(device))
                else:
                    coord_list_tensor.append(item.to(device))
            grid = torch.cartesian_prod(*coord_list_tensor)
    elif mode == 'mat':
        grid = np.meshgrid(*coord_list, indexing='ij')
        grid = torch.tensor(np.array(grid)).to(device)
    return grid


class Plots():
    """Class for ploting solutions."""
    def __init__(self,
                 model: Union[torch.nn.Module, torch.Tensor],
                 grid: torch.Tensor,
                 mode: str,
                 tol: Union[int, float] = 0):
        """
        Args:
            model (Union[torch.nn.Module, torch.Tensor]): *mat, NN, autograd* model.
            grid (torch.Tensor): grid in (torch.cartesian_prod or torch.meshgrid) form.
            mode (str): *mat, NN, autograd*
            tol (Union[int, float], optional): penalty for casual loss. Defaults to 0.
        """
        self.model = model
        self.grid = grid
        self.mode = mode
        self.tol = tol

    def _print_nn(self, title: str):
        """
        Solution plot for NN method.

        Args:
            title (str): title
        """
        try:
            nvars_model = self.model[-1].out_features
        except:
            nvars_model = self.model.model[-1].out_features

        nparams = self.grid.shape[1]
        fig = plt.figure(figsize=(15, 8))
        for i in range(nvars_model):
            if nparams == 1:
                ax1 = fig.add_subplot(1, nvars_model, i + 1)
                if title is not None:
                    ax1.set_title(title + ' variable {}'.format(i))
                ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                            self.model(self.grid)[:, i].detach().cpu().numpy())

            else:
                ax1 = fig.add_subplot(1, nvars_model, i + 1, projection='3d')
                if title is not None:
                    ax1.set_title(title + ' variable {}'.format(i))

                if self.tol != 0:
                    ax1.plot_trisurf(self.grid[:, 1].detach().cpu().numpy(),
                                     self.grid[:, 0].detach().cpu().numpy(),
                                     self.model(self.grid)[:, i].detach().cpu().numpy(),
                                     cmap=cm.jet, linewidth=0.2, alpha=1)
                else:
                    ax1.plot_trisurf(self.grid[:, 0].detach().cpu().numpy(),
                                     self.grid[:, 1].detach().cpu().numpy(),
                                     self.model(self.grid)[:, i].detach().cpu().numpy(),
                                     cmap=cm.jet, linewidth=0.2, alpha=1)
                ax1.set_xlabel("x1")
                ax1.set_ylabel("x2")

    def _print_mat(self, title: str):
        """
        Solution plot for mat method.

        Args:
           title (str): title
        """

        nparams = self.grid.shape[0]
        nvars_model = self.model.shape[0]
        fig = plt.figure(figsize=(15,8))
        for i in range(nvars_model):
            if nparams == 1:
                ax1 = fig.add_subplot(1, nvars_model, i+1)
                if title is not None:
                    ax1.set_title(title+' variable {}'.format(i))
                ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                            self.model[i].detach().cpu().numpy().reshape(-1))
            else:
                ax1 = fig.add_subplot(1, nvars_model, i+1, projection='3d')

                if title is not None:
                    ax1.set_title(title+' variable {}'.format(i))
                ax1.plot_trisurf(self.grid[0].detach().cpu().numpy().reshape(-1),
                            self.grid[1].detach().cpu().numpy().reshape(-1),
                            self.model[i].detach().cpu().numpy().reshape(-1),
                            cmap=cm.jet, linewidth=0.2, alpha=1)
            ax1.set_xlabel("x1")
            ax1.set_ylabel("x2")

    def _dir_path(self, save_dir: str) -> str:
        """ Path for save figures.

        Args:
            save_dir (str): directory where saves in

        Returns:
            str: directory where saves in
        """

        if save_dir is None:
            try:
                img_dir = os.path.join(os.path.dirname(__file__), 'img')
            except:
                current_dir = globals()['_dh'][0]
                img_dir = os.path.join(os.path.dirname(current_dir), 'img')

            if not os.path.isdir(img_dir):
                os.mkdir(img_dir)
            directory = os.path.abspath(os.path.join(img_dir,
                                        str(datetime.datetime.now().timestamp()) + '.png'))
        else:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            directory = os.path.join(save_dir,
                                     str(datetime.datetime.now().timestamp()) + '.png')
        return directory

    def solution_print(
        self,
        title: str = None,
        solution_print: bool = False,
        solution_save: bool = False,
        save_dir: str = None):
        """ printing figures.

        Args:
            title (str, optional): title of figure. Defaults to None.
            solution_print (bool, optional): print ot not figure during solving. Defaults to False.
            solution_save (bool, optional): save ot not figure during solving. Defaults to False.
            save_dir (str, optional): path to save figure. Defaults to None.
        """

        directory = self._dir_path(save_dir)

        if self.mode == 'mat':
            self._print_mat(title)
        else:
            self._print_nn(title)
        if solution_save:
            plt.savefig(directory)
        if solution_print:
            plt.show()
        plt.close()


class Solver():
    """
    High-level interface for solving equations.
    """

    def __init__(
        self,
        grid: torch.Tensor,
        equal_cls: Any,
        model: Union[torch.Tensor, torch.nn.Module],
        mode: str,
        weak_form: Union[None, list] = None):
        """
        Args:
            grid (torch.Tensor): grid in (torch.cartesian_prod or torch.meshgrid) form.
            equal_cls (Any): Equation_{NN, mat, autograd} object.
            model (Union[torch.Tensor, torch.nn.Module]): *mat, NN, autograd* model.
            mode (str): *mat, NN, autograd*, equation solving way.
            weak_form (Union[None, list], optional): list with basis functions,
            if the form is *weak*. Defaults to None.
        """

        self.grid = check_device(grid)
        self.equal_cls = equal_cls
        self.model = model
        self.mode = mode
        self.weak_form = weak_form
        self.t = 0
        self.stop_dings = 0
        self.t_imp_start = 0
        self.device = device_type()
        self.check = None
        self.inverse_param = None
        self.line = None

    def optimizer_choice(
        self,
        optimizer: Union[str, Any],
        learning_rate: float) -> \
            Union[torch.optim.Adam, torch.optim.SGD, torch.optim.LBFGS, PSO]:
        """ Setting optimizer. If optimizer is string type, it will get default settings,
            or it may be custom optimizer defined by user.

        Args:
           optimizer: optimizer choice (Adam, SGD, LBFGS, PSO).
           learning_rate: determines the step size at each iteration
           while moving toward a minimum of a loss function.

        Returns:
            optimzer: ready optimizer.
        """

        if optimizer == 'Adam':
            torch_optim = torch.optim.Adam
        elif optimizer == 'SGD':
            torch_optim = torch.optim.SGD
        elif optimizer == 'LBFGS':
            torch_optim = torch.optim.LBFGS
        elif optimizer == 'PSO':
            optimizer = PSO()
            optimizer.param_init(self.sln_cls)
            return optimizer
        else:
            try:
                optimizer.param_init(self.sln_cls)
                print('Custom optimizer is activated')
            except:
                None
            return optimizer

        if self.mode in ('NN', 'autograd'):
            optimizer = torch_optim(self.model.parameters(), lr=learning_rate)
        elif self.mode == 'mat':
            optimizer = torch_optim([self.model.requires_grad_()], lr=learning_rate)

        return optimizer

    def str_param(self):
        """Print the coefficients determining during solution.
        (for inverse tasks)
        """
        if self.inverse_parameters is not None:
            param = list(self.inverse_param.keys())
            for name, p in self.model.named_parameters():
                if name in param:
                    try:
                        param_str += name + '=' + str(p.item()) + ' '
                    except:
                        param_str = name + '=' + str(p.item()) + ' '
            print(param_str)

    def line_create(self, loss_oscillation_window: int):
        """ Approximating last_loss list (len(last_loss)=loss_oscillation_window) by the line.

        Args:
            loss_oscillation_window (int): length of last_loss list.
        """
        self.line = np.polyfit(range(loss_oscillation_window), self.last_loss, 1)

    def window_check(self, eps: float, loss_oscillation_window: int):
        """ Stopping criteria. We devide angle coeff of the approximating
        line (line_create()) on current loss value and compare one with *eps*

        Args:
            eps (float): min value for stopping criteria.
            loss_oscillation_window (int): list of losses length.
        """
        if self.t % loss_oscillation_window == 0 and self.check is None:
            self.line_create(loss_oscillation_window)
            if abs(self.line[0] / self.cur_loss) < eps and self.t > 0:
                self.stop_dings += 1
                if self.mode == 'NN' or self.mode == 'autograd':
                    self.model.apply(self.r)
                self.check = 'window_check'

    def patience_check(self, no_improvement_patience: int):
        """ Stopping criteria. We control the minimum loss and count steps
        when the current loss is bigger then min_loss. If these steps equal to
        no_improvement_patience parameter, the stopping criteria will be achieved.

        Args:
            no_improvement_patience (int): no improvement steps param.
        """
        if (self.t - self.t_imp_start) == no_improvement_patience and self.check is None:
            self.t_imp_start = self.t
            self.stop_dings += 1
            if self.mode == 'NN' or self.mode == 'autograd':
                self.model.apply(self.r)
            self.check = 'patience_check'

    def absloss_check(self, abs_loss: float):
        """ Stopping criteria. If current loss absolute value is lower then *abs_loss* param,
        the stopping criteria will be achieved.

        Args:
            abs_loss (float): stopping parameter.
        """
        if abs_loss is not None and self.cur_loss < abs_loss and self.check is None:
            self.stop_dings += 1

            self.check = 'absloss_check'

    def info_string(self):
        """ Print info string containing loss info and stop dings info.
        """

        loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
        info = 'Step = {} loss = {:.6f} normalized loss line= {:.6f}x+{:.6f}. There was {} stop dings already.'.format(
                    self.t, loss, self.line[0] / loss, self.line[1] / loss, self.stop_dings + 1)
        print(info)

    def verbose_print(
        self,
        no_improvement_patience: int,
        print_every: Union[None, int]):
        """

        Args:
            no_improvement_patience (int): no improvement steps param. (see patience_check())
            print_every (Union[None, int]): print or save after *print_every* steps.
        """

        if self.check == 'window_check':
            print('[{}] Oscillation near the same loss'.format(
                            datetime.datetime.now()))
        elif self.check == 'patience_check':
            print('[{}] No improvement in {} steps'.format(
                        datetime.datetime.now(), no_improvement_patience))
        elif self.check == 'absloss_check':
            print('[{}] Absolute value of loss is lower than threshold'.format(
                                                        datetime.datetime.now()))

        if print_every is not None and (self.t % print_every == 0):
            self.check = 'print_every'
            print('[{}] Print every {} step'.format(datetime.datetime.now(), print_every))

        if self.check is not None:
            self.info_string()
            self.str_param()
            self.plot.solution_print(title='Iteration = ' + str(self.t),
                                                solution_print=self.step_plot_print,
                                                solution_save=self.step_plot_save,
                                                save_dir=self.image_save_dir)

        self.check = None

    def amp_mixed(self, mixed_precision: bool):
        """ Preparation for mixed precsion operations.

        Args:
            mixed_precision (bool): use or not torch.amp.

        Raises:
            NotImplementedError: AMP and the LBFGS optimizer are not compatible.

        Returns:
            scaler: GradScaler for CUDA.
            cuda_flag (bool): True, if CUDA is activated.
            dtype (dtype): operations dtype.
        """

        if mixed_precision:
            scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
            print(f'Mixed precision enabled. The device is {self.device}')
            if self.optimizer.__class__.__name__ == "LBFGS":
                raise NotImplementedError("AMP and the LBFGS optimizer are not compatible.")
        else:
            scaler = None
        cuda_flag = True if self.device == 'cuda' and mixed_precision else False
        dtype = torch.float16 if self.device == 'cuda' else torch.bfloat16

        return scaler, cuda_flag, dtype

    def optimizer_step(
        self,
        mixed_precision,
        second_order_interactions,
        sampling_N,
        lambda_update,
        normalized_loss_stop):
        
        scaler, cuda_flag, dtype = self.amp_mixed(mixed_precision)

        def closure():
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device, dtype=dtype, enabled=mixed_precision):
                loss, loss_normalized = self.sln_cls.evaluate(second_order_interactions,
                                                              sampling_N, lambda_update)

            loss.backward()
            self.cur_loss = loss_normalized.item() if normalized_loss_stop else loss.item()
            return loss

        def closure_cuda():
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device, dtype=dtype, enabled=mixed_precision):
                loss, loss_normalized = self.sln_cls.evaluate(second_order_interactions,
                                                              sampling_N, lambda_update)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            self.cur_loss = loss_normalized.item() if normalized_loss_stop else loss.item()
            return loss

        try:
            self.optimizer.name == 'PSO'
            self.cur_loss = self.optimizer.step()
        except:
            self.optimizer.step(closure) if not cuda_flag else closure_cuda()

    def model_save(self, cache_utils, save_always, scaler, name):
        if save_always:
            if self.mode == 'mat':
                cache_utils.save_model_mat(model=self.model, grid=self.grid, name=name)
            else:
                scaler = scaler if scaler else None
                cache_utils.save_model(model=self.model, optimizer=self.optimizer,
                                       scaler=scaler, name=name)

    def solve(self,
              lambda_operator: Union[float, list] = 1, lambda_bound: Union[float, list] = 10,
              derivative_points: float = 2, lambda_update: bool = False, second_order_interactions: bool = True,
              sampling_N: int = 1, verbose: int = 0,
              learning_rate: float = 1e-4, gamma: float = None, lr_decay: int = 1000,
              eps: float = 1e-5, tmin: int = 1000, tmax: float = 1e5,
              nmodels: Union[int, None] = None, name: Union[str, None] = None,
              abs_loss: Union[None, float] = None, use_cache: bool = True,
              cache_dir: str = '../cache/', cache_verbose: bool = False,
              save_always: bool = False, print_every: Union[int, None] = 100,
              cache_model: Union[torch.nn.Sequential, None] = None,
              patience: int = 5, loss_oscillation_window: int = 100,
              no_improvement_patience: int = 1000, model_randomize_parameter: Union[int, float] = 0,
              optimizer_mode: str = 'Adam', step_plot_print: Union[bool, int] = False,
              step_plot_save: Union[bool, int] = False, image_save_dir: Union[str, None] = None, tol: float = 0,
              clear_cache: bool = False, normalized_loss_stop: bool = False, inverse_parameters: dict = None,
              mixed_precision: bool = False) -> Any:
        """
        High-level interface for solving equations.

        Args:
            lambda_operator: coeff for operator part in loss.
            lambda_bound: coeff for boundary part in loss.
            lambda_update: enable lambda update.
            verbose: detailed info about training process.
            learning_rate: determines the step size at each iteration while moving toward a minimum of a loss function.
            gamma: multiplicative factor of learning rate decay.
            lr_decay: decays the learning rate of each parameter group by gamma every epoch.
            eps: arbitrarily small number that uses for loss comparison criterion.
            tmax: maximum execution time.
            nmodels: number cached models
            name: model name if saved.
            abs_loss: absolute loss.
            use_cache: as is.
            cache_dir: directory where saved cache in.
            cache_verbose: detailed info about models in cache.
            save_always: saves trained model even if the cache is False.
            print_every: prints the state of each given iteration to the command line.
            cache_model: model that uses in cache
            patience:if the loss is less than a certain value, then the counter increases when it reaches the given patience, the calculation stops.
            loss_oscillation_window: smth
            no_improvement_patience: smth
            model_randomize_parameter: creates a random model parameters (weights, biases) multiplied with a given randomize parameter.
            optimizer_mode: optimizer choice (Adam, SGD, LBFGS).
            step_plot_print: draws a figure through each given step.
            step_plot_save: saves a figure through each given step.
            image_save_dir: a directory where saved figure in.
            tol: float constant, influences on error penalty in casual_loss algorithm.
            derivative_points:
            sampling_N:

        Returns:
            model.
        """
        """
        Cache initialization.
        """

        self.verbose = verbose
        self.r = create_random_fn(model_randomize_parameter)
        self.inverse_parameters = inverse_parameters
        self.step_plot_save = step_plot_save
        self.step_plot_print = step_plot_print
        self.image_save_dir = image_save_dir
        self.patience = patience
        scaler, _, dtype = self.amp_mixed(mixed_precision)

        cache_utils = CacheUtils()
        if use_cache:
            cache_utils.cache_dir = cache_dir
            cache_cls = Cache(self.grid, self.equal_cls, self.model, self.mode, self.weak_form, mixed_precision)
            self.model = cache_cls.cache(nmodels,
                                         lambda_operator,
                                         lambda_bound,
                                         cache_verbose,
                                         model_randomize_parameter,
                                         cache_model,
                                         return_normalized_loss=normalized_loss_stop)
        if clear_cache:
            cache_utils.clear_cache_dir()

        self.sln_cls = Solution(self.grid, self.equal_cls,
                           self.model, self.mode, self.weak_form,
                           lambda_operator, lambda_bound, tol, derivative_points)
        with torch.autocast(device_type=self.device, dtype=dtype, enabled=mixed_precision):
            min_loss, _ = self.sln_cls.evaluate()

        self.last_loss = np.zeros(loss_oscillation_window) + float(min_loss)
        self.cur_loss = min_loss

        self.optimizer = self.optimizer_choice(optimizer_mode, learning_rate)

        self.plot = Plots(self.model, self.grid, self.mode, tol)

        if gamma != None:
            scheduler = ExponentialLR(self.optimizer, gamma=gamma)

        if verbose:
            print('[{}] initial (min) loss is {}'.format(
                datetime.datetime.now(), min_loss.item()))

        while self.stop_dings <= self.patience or self.t < tmin:
            self.optimizer_step(mixed_precision, second_order_interactions,
                                sampling_N, lambda_update, normalized_loss_stop)

            if self.cur_loss != self.cur_loss:
                print(f'Loss is equal to NaN, something went wrong (LBFGS+high'
                        f'learning rate and pytorch<1.12 could be the problem)')
                break

            self.last_loss[(self.t - 1) % loss_oscillation_window] = self.cur_loss

            if self.cur_loss < min_loss:
                min_loss = self.cur_loss
                self.t_imp_start = self.t

            if gamma != None and self.t % lr_decay == 0:
                scheduler.step()

            self.window_check(eps, loss_oscillation_window)

            self.patience_check(no_improvement_patience)

            self.absloss_check(abs_loss)

            if verbose:
                self.verbose_print(no_improvement_patience, print_every)

            self.t += 1
            if self.t > tmax:
                break
        
        self.model_save(cache_utils, save_always, scaler, name)
        
        return self.model
