import itertools
import os
import datetime
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from scipy.interpolate import griddata
from tedeous.callbacks.callback import Callback


class Plots(Callback):
    """Class for ploting solutions."""

    def __init__(self,
                 print_every: Union[int, None] = 500,
                 save_every: Union[int, None] = 500,
                 title: str = None,
                 img_dir: str = None,
                 img_dim: str = None,
                 scatter_flag: bool = False,
                 plot_axes: List[int] = None,
                 fixed_axes: List[int] = None,
                 n_samples: int = 1,
                 img_rows: int = None,
                 img_cols: int = None,
                 figsize: tuple = (15, 8)):
        """
        Args:
            print_every (Union[int, None], optional): print plots after every *print_every* steps. Defaults to 500.
            save_every (Union[int, None], optional): save plots after every *print_every* steps. Defaults to 500.
            title (str, optional): plots title. Defaults to None.
            img_dir (str, optional): directory title where plots are being saved. Defaults to None.
        """
        super().__init__()
        self.print_every = print_every if print_every is not None else 0.1
        self.save_every = save_every if save_every is not None else 0.1
        self.nvars_model = None
        self.title = title
        self.img_dir = img_dir
        self.img_dim = img_dim
        self.plot_axes = plot_axes
        self.fixed_axes = fixed_axes
        self.n_samples = n_samples
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.scatter_flag = scatter_flag
        self.figsize = figsize
        self.attributes = {'model': ['out_features', 'output_dim', 'width_out'],
                           'layers': ['out_features', 'output_dim', 'width_out']}

    def _init_nvars_model(self):
        nvars_model = None

        for key, values in self.attributes.items():
            for value in values:
                try:
                    nvars_model = getattr(getattr(self.net, key)[-1], value)
                    break
                except AttributeError:
                    pass

        if nvars_model is None:
            try:
                return self.net[-1].out_features
            except:
                return self.net.width_out[-1]

    def _plot_img_2d(self, i_ax, j_ax, axes, nparams, fixed_values=None):
        """
        Solution plot in 2-nd dimension.
        """
        if nparams == 1:
            ax = axes[i_ax][j_ax]
            if self.title is not None:
                ax.set_title(f'{self.title} variable {i_ax}')
            result_img = ax.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                                    self.net(self.grid)[:, j_ax].detach().cpu().numpy())
        else:
            if self.fixed_axes is not None:
                for axis, value in zip(self.fixed_axes, fixed_values):
                    mask = self.grid[:, axis] == value
                subgrid = self.grid[mask]
            else:
                subgrid = self.grid

            ax = axes[i_ax][j_ax]

            grid_x = subgrid[:, self.plot_axes[0]].detach().cpu().numpy()
            grid_y = subgrid[:, self.plot_axes[1]].detach().cpu().numpy()
            if self.nvars_model == 1:
                u_values = self.net(subgrid)[:, 0].detach().cpu().numpy()
            else:
                if nparams == 2:
                    u_values = self.net(subgrid)[:, j_ax].detach().cpu().numpy()
                else:
                    u_values = self.net(subgrid)[:, i_ax].detach().cpu().numpy()

            if self.scatter_flag:
                result_img = ax.scatter(grid_x, grid_y, c=u_values, cmap=cm.jet, s=5, alpha=0.7)
            else:
                axes_size = int(self.grid.shape[0] ** 0.5)

                xi = np.linspace(grid_x.min(), grid_x.max(), axes_size)
                yi = np.linspace(grid_y.min(), grid_y.max(), axes_size)
                xi, yi = np.meshgrid(xi, yi)
                ui = griddata((grid_x, grid_y), u_values, (xi, yi), method='cubic')

                result_img = ax.imshow(ui, extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
                                       origin='lower', cmap=cm.jet, aspect='auto')

            if nparams > 2:
                if self.nvars_model > 1:
                    ax.set_title(
                        f"variable_{i_ax + 1}(x{self.plot_axes[0] + 1}, x{self.plot_axes[1] + 1}) fixed at "
                        f"{'; '.join([f'x{key + 1} = {value} ' for key, value in dict(zip(self.fixed_axes, fixed_values)).items()])}")
                else:
                    ax.set_title(
                        f"fixed at "
                        f"{'; '.join([f'x{key + 1} = {value} ' for key, value in dict(zip(self.fixed_axes, fixed_values)).items()])}")
            else:
                if self.title is not None:
                    ax.set_title(f'{self.title} variable_{i_ax}')

            ax.set_xlabel(f"x{self.plot_axes[0] + 1}")
            ax.set_ylabel(f"x{self.plot_axes[1] + 1}")

        return result_img, ax

    def _plot_img_3d(self, i_ax, j_ax, axes, nparams, fixed_values=None):
        """
        Solution plot in 3-rd dimension.
        """
        if self.fixed_axes is not None:
            for axis, value in zip(self.fixed_axes, fixed_values):
                mask = self.grid[:, axis] == value
            subgrid = self.grid[mask]
        else:
            subgrid = self.grid

        ax = axes[i_ax][j_ax]

        grid_x = subgrid[:, self.plot_axes[0]].detach().cpu().numpy()
        grid_y = subgrid[:, self.plot_axes[1]].detach().cpu().numpy()
        if self.nvars_model == 1:
            u_values = self.net(subgrid)[:, 0].detach().cpu().numpy()
        else:
            if nparams == 2:
                u_values = self.net(subgrid)[:, j_ax].detach().cpu().numpy()
            else:
                u_values = self.net(subgrid)[:, i_ax].detach().cpu().numpy()

        if self.scatter_flag:
            ax.scatter(grid_x, grid_y, u_values, c=u_values, cmap=cm.jet, s=5, alpha=0.8)
        else:
            ax.plot_trisurf(grid_x, grid_y, u_values, cmap=cm.jet, linewidth=0.2, alpha=1)

        if nparams > 2:
            if self.nvars_model > 1:
                ax.set_title(
                    f"variable_{i_ax + 1}(x{self.plot_axes[0] + 1}, x{self.plot_axes[1] + 1}) fixed at "
                    f"{'; '.join([f'x{key + 1} = {value} ' for key, value in dict(zip(self.fixed_axes, fixed_values)).items()])}")
            else:
                ax.set_title(
                    f"fixed at "
                    f"{'; '.join([f'x{key + 1} = {value} ' for key, value in dict(zip(self.fixed_axes, fixed_values)).items()])}")
        else:
            if self.title is not None:
                ax.set_title(f'{self.title} variable_{i_ax}')

        ax.set_xlabel(f"x{self.plot_axes[0] + 1}")
        ax.set_ylabel(f"x{self.plot_axes[1] + 1}")

    def _plot_img_4d(self, i_ax, j_ax, axes, nparams, fixed_values=None):
        """
        Solution plot in 4-th dimension.
        """
        if self.fixed_axes is not None:
            for axis, value in zip(self.fixed_axes, fixed_values):
                mask = self.grid[:, axis] == value
            subgrid = self.grid[mask]
        else:
            subgrid = self.grid

        ax = axes[i_ax][j_ax]

        grid_x = subgrid[:, self.plot_axes[0]].detach().cpu().numpy()
        grid_y = subgrid[:, self.plot_axes[1]].detach().cpu().numpy()
        grid_z = subgrid[:, self.plot_axes[2]].detach().cpu().numpy()
        u_values = self.net(subgrid)[:, i_ax].detach().cpu().numpy()

        ax.scatter(grid_x, grid_y, grid_z, c=u_values.flatten(), cmap=cm.jet, s=10, alpha=0.8)

        if nparams > 2:
            ax.set_title(
                f"function_{i_ax + 1}(x{self.plot_axes[0] + 1}, x{self.plot_axes[1] + 1}), "
                f"{[f'Axis {key}: {value} ' for key, value in dict(zip(self.fixed_axes, fixed_values)).items()]}")
        else:
            if self.title is not None:
                ax.set_title(f'{self.title} variable {i_ax}')

        if self.plot_axes is None:
            self.plot_axes = [0, 1, 2]

        ax.set_xlabel(f"x{self.plot_axes[0] + 1}")
        ax.set_ylabel(f"x{self.plot_axes[1] + 1}")
        ax.set_zlabel(f"x{self.plot_axes[2] + 1}")

    def _print_mat(self):
        """
        Solution plot for mat mode.
        """

        nparams = self.grid.shape[0]
        nvars_model = self.net.shape[0]
        fig = plt.figure(figsize=(15, 8))

        for i in range(nvars_model):
            if nparams == 1:
                ax1 = fig.add_subplot(1, nvars_model, i + 1)
                if self.title is not None:
                    ax1.set_title(self.title + ' variable {}'.format(i))
                ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                            self.net[i].detach().cpu().numpy().reshape(-1))
            else:
                ax1 = fig.add_subplot(1, nvars_model, i + 1, projection='3d')

                if self.title is not None:
                    ax1.set_title(self.title + ' variable {}'.format(i))
                ax1.plot_trisurf(self.grid[0].detach().cpu().numpy().reshape(-1),
                                 self.grid[1].detach().cpu().numpy().reshape(-1),
                                 self.net[i].detach().cpu().numpy().reshape(-1),
                                 cmap=cm.jet, linewidth=0.2, alpha=1)
            ax1.set_xlabel("x1")
            ax1.set_ylabel("x2")

    def _dir_path(self, save_dir: str, suffix: str = "") -> str:
        """ Path for save figures.

        Args:
            save_dir (str): directory where saves in
            suffix (str): suffix for file name

        Returns:
            str: directory where saves in
        """
        if save_dir is None:
            try:
                img_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'img')
            except:
                current_dir = globals()['_dh'][0]
                img_dir = os.path.join(os.path.dirname(current_dir), 'img')

            if not os.path.isdir(img_dir):
                os.mkdir(img_dir)
            directory = os.path.abspath(os.path.join(img_dir,
                                                     str(datetime.datetime.now().timestamp()) + suffix + '.png'))
        else:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            directory = os.path.join(save_dir,
                                     str(datetime.datetime.now().timestamp()) + suffix + '.png')
        return directory

    def solution_print(self):
        """ printing or saving figures.
        """

        print_flag = self.model.t % self.print_every == 0
        save_flag = self.model.t % self.save_every == 0

        if print_flag or save_flag:
            self.net = self.model.net
            self.grid = self.model.solution_cls.grid

            if self.model.mode == 'mat':
                self._print_mat()
            else:
                self.nvars_model = self._init_nvars_model()
                nparams = self.grid.shape[1]

                if self.img_dim is None:
                    if nparams + 1 <= 4:
                        self.img_dim = f"{nparams + 1}d"
                    else:
                        self.img_dim = '2d'

                if self.img_rows is None:
                    if nparams > 2:
                        self.img_rows = self.nvars_model
                    else:
                        self.img_rows = 1

                if self.img_cols is None:
                    if nparams > 2:
                        self.img_cols = self.n_samples
                    else:
                        self.img_cols = self.nvars_model

                if self.plot_axes is None:
                    self.plot_axes = [0, 1]

                if self.fixed_axes is None:
                    fixed_points_combinations = range(self.img_cols)
                else:
                    fixed_points = []
                    for axis in self.fixed_axes:
                        unique_values = torch.unique(self.grid[:, axis]).detach().cpu().numpy()
                        selected_idx = torch.linspace(0, len(unique_values) - 1,
                                                      self.n_samples).long().detach().cpu().numpy()
                        selected_values = unique_values[selected_idx] if len(
                            unique_values) >= self.n_samples else unique_values
                        fixed_points.append(selected_values)

                    fixed_points_combinations = list(itertools.product(*fixed_points))

                if self.img_dim == '2d':
                    fig, axes = plt.subplots(self.img_rows,
                                             self.img_cols,
                                             figsize=(self.figsize[0] * self.img_cols, self.figsize[1] * self.img_rows),
                                             squeeze=False)
                elif self.img_dim == '3d' or self.img_dim == '4d':
                    fig, axes = plt.subplots(self.img_rows,
                                             self.img_cols,
                                             figsize=(self.figsize[0] * self.img_cols, self.figsize[1] * self.img_rows),
                                             subplot_kw={'projection': '3d'},
                                             squeeze=False)

                if nparams > 1:
                    i_fix_value = 0

                for i_ax in range(self.img_rows):
                    if self.nvars_model > 1:
                        i_fix_value = 0
                    for j_ax in range(self.img_cols):
                        if nparams > 1:
                            fixed_values = fixed_points_combinations[i_fix_value]
                            i_fix_value += 1
                        else:
                            fixed_values = None
                        if self.img_dim == '2d':
                            result_img, ax = self._plot_img_2d(i_ax, j_ax, axes, nparams, fixed_values=fixed_values)
                            if nparams > 1:
                                fig.colorbar(result_img, ax=ax)
                        elif self.img_dim == '3d':
                            self._plot_img_3d(i_ax, j_ax, axes, nparams, fixed_values=fixed_values)
                        elif self.img_dim == '4d':
                            self._plot_img_4d(i_ax, j_ax, axes, nparams, fixed_values=fixed_values)

                if save_flag:
                    directory = self._dir_path(self.img_dir)
                    plt.savefig(directory)
                if print_flag:
                    plt.show()

                plt.close()

    def on_epoch_end(self, logs=None):
        self.solution_print()

