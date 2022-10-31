import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import input_preprocessing
from cache import Model_prepare
from typing import Union
import os
import sys
import datetime


def grid_format_prepare(coord_list: Union[torch.Tensor, list, np.ndarray], mode: str = 'NN') -> torch.Tensor:
    """
    Prepares grid to a general form. Further, formatted grid can be processed
    by Points_type.point_typization for 'NN' and 'autograd' methods.

    Parameters
    ----------
    coord_list
        list with coordinates.
    mode
        Calculation method. (i.e., "NN", "autograd", "mat")

    Returns
    -------
    grid
        grid in a general form
    """
    if type(coord_list) == torch.Tensor:
        print('Grid is a tensor, assuming old format, no action performed')
        return coord_list
    if mode == 'NN' or mode == 'autograd':
        if len(coord_list) == 1:
            coord_list = torch.tensor(coord_list)
            grid = coord_list.reshape(-1, 1).float()
        else:
            coord_list_tensor = []
            for item in coord_list:
                if isinstance(item, (np.ndarray)):
                    coord_list_tensor.append(torch.from_numpy(item))
                else:
                    coord_list_tensor.append(item)
            grid = torch.cartesian_prod(*coord_list_tensor).float()
    elif mode == 'mat':
        grid = np.meshgrid(*coord_list)
        grid = torch.tensor(np.array(grid))
    return grid


class Solver(Model_prepare):
    """
    High-level interface for solving equations.
    """
    def __init__(self, grid: torch.Tensor, equal_cls: Union[input_preprocessing.Equation_NN,
                input_preprocessing.Equation_mat, input_preprocessing.Equation_autograd],
                 model: torch.nn.Sequential, mode: str):
        """
        High-level interface for solving equations.

        Parameters
        ----------
        grid
            array of a n-D points.
        equal_cls
            object from input_preprocessing (see input_preprocessing.Equation).
        model
            neural network.
        mode
            Calculation method. (i.e., "NN", "autograd", "mat").
        """
        super().__init__(grid, equal_cls, model, mode)

    def optimizer_choice(self, optimizer: str, learning_rate: float) -> \
            Union[torch.optim.Adam, torch.optim.SGD, torch.optim.LBFGS]:
        """
        Parameters
        ----------
        optimizer:
            optimizer choice (Adam, SGD, LBFGS).
        learning_rate:
            determines the step size at each iteration while moving toward a minimum of a loss function.

        Returns
        -------
        optimizer
            torch.optimizer object as is.

        """
        if optimizer == 'Adam':
            if self.mode == 'NN' or self.mode == 'autograd':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            elif self.mode == 'mat':
                optimizer = torch.optim.Adam([self.model.requires_grad_()], lr=learning_rate)

        elif optimizer == 'SGD':
            if self.mode == 'NN' or self.mode == 'autograd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
            elif self.mode == 'mat':
                optimizer = torch.optim.SGD([self.model.requires_grad_()], lr=learning_rate)

        elif optimizer == 'LBFGS':
            if self.mode == 'NN' or self.mode == 'autograd':
                optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate)
            elif self.mode == 'mat':
                optimizer = torch.optim.LBFGS([self.model.requires_grad_()], lr=learning_rate)

        else:
            print('Wrong optimizer chosen, optimization was not performed')
            return self.model

        return optimizer

    def solution_print(self, title: Union[str, None] = None, solution_print: bool = False,
                       solution_save: bool = True, save_dir: Union[str, None] = None):
        """
        Visualizes the resulting solution.

        Parameters
        ----------
        title
            as is.
        solution_print
            draws a figure.
        solution_save:
            saves figure.
        save_dir:
            a directory where saved figure in.
        """
        if save_dir == None:
            img_dir = os.path.join(os.path.dirname(__file__), 'img')
            if not (os.path.isdir(img_dir)):
                os.mkdir(img_dir)
            directory = os.path.abspath(os.path.join(img_dir, str(datetime.datetime.now().timestamp()) + '.png'))
        else:
            directory = os.path.join(save_dir, str(datetime.datetime.now().timestamp()) + '.png')
        if self.mode == 'NN' or self.mode == 'autograd':
            nvars_model = self.model(self.grid).shape[-1]
            nparams = self.grid.shape[1]
            fig = plt.figure()
            for i in range(nvars_model):
                if nparams == 1:
                    ax1 = fig.add_subplot(1, nvars_model, i + 1)
                    if title != None:
                        ax1.set_title(title + ' variable {}'.format(i))
                    ax1.scatter(self.grid.detach().numpy().reshape(-1),
                                self.model(self.grid).detach().numpy().reshape(-1))
                else:
                    ax1 = fig.add_subplot(1, nvars_model, i + 1, projection='3d')

                    if title != None:
                        ax1.set_title(title + ' variable {}'.format(i))

                    ax1.plot_trisurf(self.grid[:, 0].detach().numpy().reshape(-1),
                                     self.grid[:, 1].detach().numpy().reshape(-1),
                                     self.model(self.grid)[:, i].detach().numpy().reshape(-1), cmap=cm.jet,
                                     linewidth=0.2, alpha=1)
                    ax1.set_xlabel("x1")
                    ax1.set_ylabel("x2")
            if solution_print:
                plt.show()
            if solution_save:
                plt.savefig(directory)
            plt.close()
        elif self.mode == 'mat':
            nparams = self.grid.shape[0]

            if nparams == 1:
                fig = plt.figure()
                plt.scatter(self.grid.reshape(-1), self.model.detach().numpy().reshape(-1))
                if solution_print:
                    plt.show()
                if solution_save:
                    plt.savefig(directory)
                plt.close()
            elif nparams == 2:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                if title != None:
                    ax.set_title(title)
                ax.plot_trisurf(self.grid[0].reshape(-1), self.grid[1].reshape(-1),
                                self.model.reshape(-1).detach().numpy(), cmap=cm.jet, linewidth=0.2, alpha=1)
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                if solution_print:
                    plt.show()
                if solution_save:
                    plt.savefig(directory)
                plt.close()

    def solve(self, lambda_bound: Union[int, float] = 10, verbose: bool = False, learning_rate: float = 1e-4,
              eps: float = 1e-5, tmin: int = 1000, tmax: float = 1e5, nmodels: Union[int, None] = None,
              name: Union[str, None] = None, abs_loss: Union[None, float] = None, use_cache: bool = True,
              cache_dir: str = '../cache/', cache_verbose: bool = False, save_always: bool = False,
              print_every: Union[int, None] = 100, cache_model: Union[torch.nn.Sequential, None] = None,
              patience: int = 5, loss_oscillation_window: int = 100, no_improvement_patience: int = 1000,
              model_randomize_parameter: Union[int,float] = 0, optimizer_mode: str = 'Adam',
              step_plot_print: Union[bool, int] = False, step_plot_save: Union[bool, int] = False,
              image_save_dir: Union[str, None] = None) -> torch.nn.Sequential:
        """
        High-level interface for solving equations.

        Parameters
        ----------
        lambda_bound
            an arbitrary chosen constant, influence only convergence speed.
        verbose
            more detailed info about training process.
        learning_rate
            determines the step size at each iteration while moving toward a minimum of a loss function.
        eps
            arbitrarily small number that uses for loss comparison criterion.
        tmax
            maximum execution time.
        nmodels
            ?
        name
            model name if saved.
        abs_loss: Union[None, float]
            absolute loss???.
        use_cache
            as is.
        cache_dir
            directory where saved cache in.
        cache_verbose
            more detailed info about models in cache.
        save_always
            ????
        print_every
            prints the state of each given iteration to the command line.
        cache_model
            model that uses in cache
        patience
            if the loss is less than a certain value, then the counter increases,
            when it reaches the given patience, the calculation stops.
        loss_oscillation_window

        no_improvement_patience

        model_randomize_parameter
            creates a random model parameters (weights, biases) multiplied with a given randomize parameter.
        optimizer_mode
            optimizer choice (Adam, SGD, LBFGS).
        step_plot_print
            draws a figure through each given step.
        step_plot_save
            saves a figure through each given step.
        image_save_dir
            a directory where saved figure in.

        Returns
        -------
        model
            neural network.


        """
        # prepare input data to uniform format 
        r = self.create_random_fn(model_randomize_parameter)
        #  use cache if needed
        if use_cache:
            self.model, min_loss = self.cache(cache_dir=cache_dir,
                                              nmodels=nmodels,
                                              lambda_bound=lambda_bound,
                                              cache_verbose=cache_verbose,
                                              model_randomize_parameter=model_randomize_parameter,
                                              cache_model=cache_model)

        optimizer = self.optimizer_choice(optimizer_mode, learning_rate)

        if True:
            # if not use_cache:
            min_loss = self.loss_evaluation(lambda_bound=lambda_bound)

        save_cache = False

        if min_loss > 0.1 or save_always:  # why 0.1?
            save_cache = True

        # standard NN stuff
        if verbose:
            print('[{}] initial (min) loss is {}'.format(datetime.datetime.now(), min_loss))

        t = 0

        last_loss = np.zeros(loss_oscillation_window) + float(min_loss)
        line = np.polyfit(range(loss_oscillation_window), last_loss, 1)

        def closure():
            nonlocal cur_loss
            optimizer.zero_grad()
            loss = self.loss_evaluation(lambda_bound=lambda_bound)

            loss.backward()
            cur_loss = loss.item()
            return loss

        stop_dings = 0
        t_imp_start = 0
        # to stop train proceduce we fit the line in the loss data
        # if line is flat enough 5 times, we stop the procedure
        cur_loss = min_loss
        while stop_dings <= patience:
            optimizer.step(closure)

            last_loss[t % loss_oscillation_window] = cur_loss

            if cur_loss < min_loss:
                min_loss = cur_loss
                t_imp_start = t

            if verbose:
                info_string = 'Step = {} loss = {:.6f} normalized loss line = {:.6f}x+{:.6f}. There was {} stop dings already.'.format(
                    t, cur_loss, line[0] / cur_loss, line[1] / cur_loss, stop_dings + 1)

            if t % loss_oscillation_window == 0:
                line = np.polyfit(range(loss_oscillation_window), last_loss, 1)
                if abs(line[0] / cur_loss) < eps and t > 0:
                    stop_dings += 1
                    if self.mode == 'NN' or self.mode == 'autograd':
                        self.model.apply(r)
                    if verbose:
                        print('[{}] Oscillation near the same loss'.format(datetime.datetime.now()))
                        print(info_string)
                        if step_plot_print or step_plot_save:
                            self.solution_print(title='Iteration = ' + str(t), solution_print=step_plot_print,
                                                solution_save=step_plot_save, save_dir=image_save_dir)

            if (t - t_imp_start == no_improvement_patience):
                if verbose:
                    print('[{}] No improvement in {} steps'.format(datetime.datetime.now(), no_improvement_patience))
                    print(info_string)
                    if step_plot_print or step_plot_save:
                        self.solution_print(title='Iteration = ' + str(t), solution_print=step_plot_print,
                                            solution_save=step_plot_save, save_dir=image_save_dir)
                t_imp_start = t
                stop_dings += 1
                if self.mode == 'NN' or self.mode == 'autograd':
                    self.model.apply(r)

            if abs_loss != None and cur_loss < abs_loss:
                if verbose:
                    print('[{}] Absolute value of loss is lower than threshold'.format(datetime.datetime.now()))
                    print(info_string)
                    if step_plot_print or step_plot_save:
                        self.solution_print(title='Iteration = ' + str(t), solution_print=step_plot_print,
                                            solution_save=step_plot_save, save_dir=image_save_dir)
                stop_dings += 1

            if print_every != None and (t % print_every == 0) and verbose:
                print('[{}] Print every {} step'.format(datetime.datetime.now(), print_every))
                print(info_string)
                if step_plot_print or step_plot_save:
                    self.solution_print(title='Iteration = ' + str(t), solution_print=step_plot_print,
                                        solution_save=step_plot_save, save_dir=image_save_dir)

            t += 1
            if t > tmax:
                break
        if (save_cache and use_cache) or save_always:
            if self.mode == 'mat':
                self.save_model_mat(cache_dir=cache_dir, name=name)
            else:
                self.save_model(self.model, self.model.state_dict(), optimizer.state_dict(), cache_dir=cache_dir,
                                name=name)
        return self.model
