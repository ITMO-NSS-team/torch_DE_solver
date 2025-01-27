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
                 var_transpose: bool = False,
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
        self.var_transpose = var_transpose
        self.scatter_flag = scatter_flag
        self.figsize = figsize
        self.attributes = {'model': ['out_features', 'output_dim', 'width_out'],
                           'layers': ['out_features', 'output_dim', 'width_out']}

    def _init_nvars_model(self):
        nvars_model = None

        if self.model.mode == 'mat':
            return self.net.shape[0]

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

    def filter_grid(self, fixed_values):
        if self.fixed_axes is not None:
            for axis, value in zip(self.fixed_axes, fixed_values):
                mask = self.grid[:, axis] == value
            return self.grid[mask]
        else:
            return self.grid

    def generate_plot_data(self, subgrid, nparams, i_ax, j_ax):
        lst_grid_axes = []
        n_plot_axes = nparams if len(self.plot_axes) is None else len(self.plot_axes)

        for i in range(n_plot_axes):
            lst_grid_axes.append(subgrid[self.plot_axes[i]].detach().cpu().numpy().reshape(-1)
                                 if self.model.mode == 'mat' else subgrid[:, self.plot_axes[i]].detach().cpu().numpy())

        if self.nvars_model == 1:
            u_values = self.net[0].detach().cpu().numpy().reshape(-1) if self.model.mode == 'mat' \
                else self.net(subgrid)[:, 0].detach().cpu().numpy()
        else:
            if nparams <= 2:
                idx_net = i_ax if self.var_transpose else j_ax
            else:
                idx_net = j_ax if self.var_transpose else i_ax

            u_values = self.net[idx_net].detach().cpu().numpy().reshape(-1) if self.model.mode == 'mat' \
                else self.net(subgrid)[:, idx_net].detach().cpu().numpy()

        return u_values, lst_grid_axes

    def set_labels(self, i_ax, j_ax, ax, nparams, fixed_values=None):
        if nparams > 2 and fixed_values is not None:
            title = f"fixed at {'; '.join([f'x{key + 1} = {value} ' for key, value in dict(zip(self.fixed_axes, fixed_values)).items()])}"
            if self.nvars_model > 1:
                var_idx = j_ax if self.var_transpose else i_ax
                title = f"variable_{var_idx + 1}(x{self.plot_axes[0] + 1}, x{self.plot_axes[1] + 1}) " + title
            ax.set_title(title)
        else:
            if self.title is not None:
                ax.set_title(f'{self.title} variable_{i_ax}')

        axis_functions = ["set_xlabel", "set_ylabel", "set_zlabel"]

        for i in range(len(self.plot_axes)):
            func = getattr(ax, axis_functions[i])
            func(f"x{self.plot_axes[i] + 1}")

    def _plot_img_2d(self, i_ax, j_ax, ax, nparams, fixed_values=None):
        """
        Solution plot in 2-nd dimension.
        """
        subgrid = self.filter_grid(fixed_values)

        if nparams == 1:
            u_values, grid = self.generate_plot_data(subgrid, nparams, i_ax, j_ax)
            grid_x = grid[0]

            if self.model.mode == 'mat':
                result_img = ax.scatter(grid_x, u_values)
            else:
                result_img = ax.scatter(grid_x.reshape(-1), u_values)
        else:
            u_values, grid = self.generate_plot_data(subgrid, nparams, i_ax, j_ax)
            grid_x, grid_y = grid

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

        self.set_labels(i_ax, j_ax, ax, nparams, fixed_values)

        return result_img, ax

    def _plot_img_3d(self, i_ax, j_ax, ax, nparams, fixed_values=None):
        """
        Solution plot in 3-rd dimension.
        """
        subgrid = self.filter_grid(fixed_values)
        u_values, grid = self.generate_plot_data(subgrid, nparams, i_ax, j_ax)
        grid_x, grid_y = grid

        if self.scatter_flag:
            ax.scatter(grid_x, grid_y, u_values, c=u_values, cmap=cm.jet, s=5, alpha=0.8)
        else:
            ax.plot_trisurf(grid_x, grid_y, u_values, cmap=cm.jet, linewidth=0.2, alpha=1)

        self.set_labels(i_ax, j_ax, ax, nparams, fixed_values)

    def _plot_img_4d(self, i_ax, j_ax, ax, nparams, fixed_values=None):
        """
        Solution plot on 4d image.
        """
        subgrid = self.filter_grid(fixed_values)
        u_values, grid = self.generate_plot_data(subgrid, nparams, i_ax, j_ax)
        grid_x, grid_y, grid_z = grid

        ax.scatter(grid_x, grid_y, grid_z, c=u_values.flatten(), cmap=cm.jet, s=10, alpha=0.8)

        self.set_labels(i_ax, j_ax, ax, nparams, fixed_values)

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

            self.nvars_model = self._init_nvars_model()
            nparams = self.grid.shape[0] if self.model.mode == 'mat' else self.grid.shape[1]

            if self.img_dim is None:
                self.img_dim = f"{nparams + 1}d" if nparams + 1 < 5 else '2d'

            if self.img_rows is None:
                self.img_rows = self.nvars_model if nparams > 2 else 1

            if self.img_cols is None:
                self.img_cols = self.n_samples if nparams > 2 else self.nvars_model

            if self.plot_axes is None:
                self.plot_axes = [i for i in range(nparams)] if int(self.img_dim[0]) < 5 else [0, 1]

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

            n_rows = self.img_cols if self.var_transpose else self.img_rows
            n_cols = self.img_rows if self.var_transpose else self.img_cols

            if self.img_dim == '2d':
                fig, axes = plt.subplots(n_rows,
                                         n_cols,
                                         figsize=(self.figsize[0] * n_cols, self.figsize[1] * n_rows),
                                         squeeze=False)
            elif self.img_dim == '3d' or self.img_dim == '4d':
                fig, axes = plt.subplots(n_rows,
                                         n_cols,
                                         figsize=(self.figsize[0] * n_cols, self.figsize[1] * n_rows),
                                         subplot_kw={'projection': '3d'},
                                         squeeze=False)
            if nparams > 2:
                i_fix_value = 0

            for i_ax in range(n_rows):
                if nparams > 2 and self.nvars_model > 1 and not self.var_transpose:
                    i_fix_value = 0
                for j_ax in range(n_cols):
                    ax = axes[i_ax][j_ax]
                    if 2 < nparams != len(self.plot_axes):
                        fixed_values = fixed_points_combinations[i_fix_value]
                        if not self.var_transpose:
                            i_fix_value += 1
                    else:
                        fixed_values = None
                    if self.img_dim == '2d':
                        result_img, ax = self._plot_img_2d(i_ax, j_ax, ax, nparams, fixed_values=fixed_values)
                        if nparams > 1:
                            fig.colorbar(result_img, ax=ax)
                    elif self.img_dim == '3d':
                        self._plot_img_3d(i_ax, j_ax, ax, nparams, fixed_values=fixed_values)
                    elif self.img_dim == '4d':
                        self._plot_img_4d(i_ax, j_ax, ax, nparams, fixed_values=fixed_values)

                if nparams > 2 and self.nvars_model > 1 and self.var_transpose:
                    i_fix_value += 1

            if save_flag:
                directory = self._dir_path(self.img_dir)
                plt.savefig(directory)
            if print_flag:
                plt.show()

            plt.close()

    def on_epoch_end(self, logs=None):
        self.solution_print()
