from filecmp import clear_cache
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import datetime
from typing import Union
from torch.optim.lr_scheduler import ExponentialLR

from tedeous.cache import *
from tedeous.device import check_device, device_type
from tedeous.solution import Solution
import tedeous.input_preprocessing


def grid_format_prepare(coord_list, mode='NN') -> torch.Tensor:
    """
    Prepares grid to a general form. Further, formatted grid can be processed
    by Points_type.point_typization for 'NN' and 'autograd' methods.
    Args:
        coord_list: list with coordinates.
        mode: Calculation method. (i.e., "NN", "autograd", "mat").

    Returns:
        grid in a general form.
    """
    device = device_type()
    if type(coord_list) == torch.Tensor:
        print('Grid is a tensor, assuming old format, no action performed')
        return check_device(coord_list)
    elif mode == 'NN' or mode == 'autograd':
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
    def __init__(self, model, grid, mode, tol = 0):
        self.model = model
        self.grid = grid
        self.mode = mode
        self.tol = tol

    def print_nn(self, title: str):
        """
        Solution plot for NN method.

        Args:
            title: title
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
                if title != None:
                    ax1.set_title(title + ' variable {}'.format(i))
                ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                            self.model(self.grid)[:, i].detach().cpu().numpy())

            else:
                ax1 = fig.add_subplot(1, nvars_model, i + 1, projection='3d')
                if title != None:
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

    def print_mat(self, title):
        """
        Solution plot for mat method.

        Args:
           title: title
        """

        nparams = self.grid.shape[0]
        nvars_model = self.model.shape[0]
        fig = plt.figure(figsize=(15,8))
        for i in range(nvars_model):
            if nparams == 1:
                ax1 = fig.add_subplot(1, nvars_model, i+1)
                if title != None:
                    ax1.set_title(title+' variable {}'.format(i))
                ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                            self.model[i].detach().cpu().numpy().reshape(-1))
            else:
                ax1 = fig.add_subplot(1, nvars_model, i+1, projection='3d')
                
                if title!=None:
                    ax1.set_title(title+' variable {}'.format(i))
                ax1.plot_trisurf(self.grid[0].detach().cpu().numpy().reshape(-1),
                            self.grid[1].detach().cpu().numpy().reshape(-1),
                            self.model[i].detach().cpu().numpy().reshape(-1),
                            cmap=cm.jet, linewidth=0.2, alpha=1)
            ax1.set_xlabel("x1")
            ax1.set_ylabel("x2")

    def dir_path(self, save_dir: str):
        """
        Path for save figures.

        Args:
            save_dir: directory where saves in
        """
        if save_dir == None:
            try:
                img_dir = os.path.join(os.path.dirname(__file__), 'img')
            except:
                current_dir = globals()['_dh'][0]
                img_dir = os.path.join(os.path.dirname(current_dir), 'img')

            if not (os.path.isdir(img_dir)):
                os.mkdir(img_dir)
            directory = os.path.abspath(os.path.join(img_dir,
                                                     str(datetime.datetime.now().timestamp()) + '.png'))
        else:
            if not (os.path.isdir(save_dir)):
                os.mkdir(save_dir)
            directory = os.path.join(save_dir,
                                     str(datetime.datetime.now().timestamp()) + '.png')
        return directory

    def solution_print(self, title=None, solution_print=False,
                       solution_save=False, save_dir=None):

        directory = self.dir_path(save_dir)

        if self.mode == 'mat':
            self.print_mat(title)
        else:
            self.print_nn(title)
        if solution_save:
            plt.savefig(directory)
        if solution_print:
            plt.show()
        plt.close()


class Solver():
    """
    High-level interface for solving equations.
    """

    def __init__(self, grid: torch.Tensor, equal_cls,
                 model: Any, mode: str, weak_form: Union[None, list] = None):
        """
        High-level interface for solving equations.

        Args:
            grid: array of a n-D points.
            equal_cls: object from input_preprocessing (see input_preprocessing.Equation).
            model: neural network.
            mode: calculation method. (i.e., "NN", "autograd", "mat").
            weak_form: list of basis functions.
        """
        self.grid = check_device(grid)
        self.equal_cls = equal_cls
        self.model = model
        self.mode = mode
        self.weak_form = weak_form

    def optimizer_choice(self, optimizer: str, learning_rate: float) -> \
            Union[torch.optim.Adam, torch.optim.SGD, torch.optim.LBFGS]:
        """
        Setting optimizer.

        Args:
           optimizer: optimizer choice (Adam, SGD, LBFGS).
           learning_rate: determines the step size at each iteration while moving toward a minimum of a loss function.
        Returns:
           torch.optimizer object as is.
        """
        if optimizer == 'Adam':
            torch_optim = torch.optim.Adam
        elif optimizer == 'SGD':
            torch_optim = torch.optim.SGD
        elif optimizer == 'LBFGS':
            torch_optim = torch.optim.LBFGS
        else:
            print('Wrong optimizer chosen, optimization was not performed')
            return self.model

        if self.mode == 'NN' or self.mode == 'autograd':
            optimizer = torch_optim(self.model.parameters(), lr=learning_rate)
        elif self.mode == 'mat':
            optimizer = torch_optim([self.model.requires_grad_()], lr=learning_rate)

        return optimizer

    def str_param(self, inverse_param: dict):
        param = list(inverse_param.keys())
        for name, p in self.model.named_parameters():
            if name in param:
                try:
                    param_str += name + '=' + str(p.item()) + ' '
                except:
                    param_str = name + '=' + str(p.item()) + ' '
        return param_str

    def solve(self,lambda_operator: Union[float, list] = 1,lambda_bound: Union[float, list] = 10, derivative_points:float=2,
              lambda_update: bool = False, second_order_interactions: bool = True, sampling_N: int = 1, verbose: int = 0,
              learning_rate: float = 1e-4, gamma=None, lr_decay=1000,
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
              clear_cache: bool  =False, normalized_loss_stop: bool = False, inverse_parameters: dict = None) -> Any:
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

        Returns:
            model.
        """

        Cache_class = Model_prepare(self.grid, self.equal_cls,
                                    self.model, self.mode, self.weak_form)

        Cache_class.change_cache_dir(cache_dir)

        # prepare input data to uniform format
        r = create_random_fn(model_randomize_parameter)

        if clear_cache:
            Cache_class.clear_cache_dir()

        #  use cache if needed
        if use_cache:
            self.model, min_loss = Cache_class.cache(nmodels,
                                                     lambda_operator,
                                                     lambda_bound,
                                                     cache_verbose,
                                                     model_randomize_parameter,
                                                     cache_model,
                                                    return_normalized_loss=normalized_loss_stop)

            Solution_class = Solution(self.grid, self.equal_cls,
                                      self.model, self.mode, self.weak_form,
                                      lambda_operator, lambda_bound, tol, derivative_points)
        else:
            Solution_class = Solution(self.grid, self.equal_cls,
                                      self.model, self.mode, self.weak_form,
                                      lambda_operator, lambda_bound, tol, derivative_points)

            min_loss , _ = Solution_class.evaluate()

        self.plot = Plots(self.model, self.grid, self.mode, tol)

        optimizer = self.optimizer_choice(optimizer_mode, learning_rate)

        if gamma != None:
            scheduler = ExponentialLR(optimizer, gamma=gamma)

        # standard NN stuff
        if verbose:
            print('[{}] initial (min) loss is {}'.format(
                datetime.datetime.now(), min_loss.item()))

        t = 0

        last_loss = np.zeros(loss_oscillation_window) + float(min_loss)
        line = np.polyfit(range(loss_oscillation_window), last_loss, 1)

        def closure():
            nonlocal cur_loss
            optimizer.zero_grad()
            loss, loss_normalized = Solution_class.evaluate(second_order_interactions=second_order_interactions,
                                           sampling_N=sampling_N,
                                           lambda_update=lambda_update)

            loss.backward()
            if normalized_loss_stop:
                cur_loss = loss_normalized.item()
            else:
                cur_loss = loss.item()
            return loss

        stop_dings = 0
        t_imp_start = 0
        # to stop train proceduce we fit the line in the loss data
        # if line is flat enough "patience" times, we stop the procedure
        cur_loss = min_loss
        while stop_dings <= patience:
            optimizer.step(closure)
            if cur_loss != cur_loss:
                print(f'Loss is equal to NaN, something went wrong (LBFGS+high'
                      f'learning rate and pytorch<1.12 could be the problem)')
                break

            last_loss[(t - 1) % loss_oscillation_window] = cur_loss

            if cur_loss < min_loss:
                min_loss = cur_loss
                t_imp_start = t

            if verbose:
                info_string = 'Step = {} loss = {:.6f} normalized loss line= {:.6f}x+{:.6f}. There was {} stop dings already.'.format(
                    t, cur_loss, line[0] / cur_loss, line[1] / cur_loss, stop_dings + 1)

            if gamma != None and t % lr_decay == 0:
                scheduler.step()

            if t % loss_oscillation_window == 0:
                line = np.polyfit(range(loss_oscillation_window), last_loss, 1)
                if abs(line[0] / cur_loss) < eps and t > 0:
                    stop_dings += 1
                    if self.mode == 'NN' or self.mode == 'autograd':
                        self.model.apply(r)
                    if verbose:
                        print('[{}] Oscillation near the same loss'.format(
                            datetime.datetime.now()))
                        print(info_string)
                        if inverse_parameters is not None:
                            print(self.str_param(inverse_parameters))
                        if step_plot_print or step_plot_save:
                            self.plot.solution_print(title='Iteration = ' + str(t),
                                                     solution_print=step_plot_print,
                                                     solution_save=step_plot_save,
                                                     save_dir=image_save_dir)

            if (t - t_imp_start) == no_improvement_patience:
                if verbose:
                    print('[{}] No improvement in {} steps'.format(
                        datetime.datetime.now(), no_improvement_patience))
                    print(info_string)
                    if inverse_parameters is not None:
                        print(self.str_param(inverse_parameters))
                    if step_plot_print or step_plot_save:
                        self.plot.solution_print(title='Iteration = ' + str(t),
                                                 solution_print=step_plot_print,
                                                 solution_save=step_plot_save,
                                                 save_dir=image_save_dir)
                t_imp_start = t
                stop_dings += 1
                if self.mode == 'NN' or self.mode == 'autograd':
                    self.model.apply(r)

            if abs_loss != None and cur_loss < abs_loss:
                if verbose:
                    print('[{}] Absolute value of loss is lower than threshold'.format(datetime.datetime.now()))
                    print(info_string)
                    if inverse_parameters is not None:
                        print(self.str_param(inverse_parameters))
                    if step_plot_print or step_plot_save:
                        self.plot.solution_print(title='Iteration = ' + str(t),
                                                 solution_print=step_plot_print,
                                                 solution_save=step_plot_save,
                                                 save_dir=image_save_dir)
                stop_dings += 1
            # print('t',t)
            if print_every != None and (t % print_every == 0) and verbose:
                print('[{}] Print every {} step'.format(
                    datetime.datetime.now(), print_every))
                print(info_string)
                if inverse_parameters is not None:
                    print(self.str_param(inverse_parameters))
                # print('loss', closure().item(), 'loss_norm', cur_loss)
                if step_plot_print or step_plot_save:
                    self.plot.solution_print(title='Iteration = ' + str(t),
                                             solution_print=step_plot_print,
                                             solution_save=step_plot_save,
                                             save_dir=image_save_dir)

            t += 1
            if t > tmax:
                break
        if save_always:
            if self.mode == 'mat':
                Cache_class.save_model_mat(name=name, cache_verbose=cache_verbose)
            else:
                Cache_class.save_model(self.model, self.model.state_dict(),
                                       optimizer.state_dict(),
                                       name=name)
        return self.model
