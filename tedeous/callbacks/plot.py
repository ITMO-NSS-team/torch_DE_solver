import os
import datetime
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from tedeous.callbacks.callback import Callback


class Plots(Callback):
    """Class for ploting solutions."""

    def __init__(self,
                 print_every: Union[int, None] = 500,
                 save_every: Union[int, None] = 500,
                 title: str = None,
                 img_dir: str = None,
                 img_dim: str = '3d',
                 t_values: list = None,
                 removed_domains: Union[List, torch.Tensor] = None):
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
        self.title = title
        self.img_dir = img_dir
        self.img_dim = img_dim
        self.t_values = t_values
        self.mask = removed_domains

        self.attributes = {'model': ['out_features', 'output_dim', 'width_out'],
                           'layers': ['out_features', 'output_dim', 'width_out']}

    def _init_nvars_model(self):
        """
        Initialization nvars_model object that describes the number of outputs.

        Returns:
            int: number of outputs
        """
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

    def _print_nn(self):
        """
        Solution plot for *NN, autograd* mode.

        """

        self.nvars_model = self._init_nvars_model()

        nparams = self.grid.shape[1]
        fig = plt.figure(figsize=(15, 8))

        for i in range(self.nvars_model):
            if self.img_dim == '3d':
                if nparams == 1:
                    ax1 = fig.add_subplot(1, self.nvars_model, i + 1)
                    if self.title is not None:
                        ax1.set_title(self.title + ' variable {}'.format(i))
                    ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                                self.net(self.grid)[:, i].detach().cpu().numpy())
                else:
                    ax1 = fig.add_subplot(1, self.nvars_model, i + 1, projection='3d')
                    if self.title is not None:
                        ax1.set_title(self.title + ' variable {}'.format(i))

                    grid_x = self.grid[:, 0].detach().cpu().numpy()
                    grid_y = self.grid[:, 1].detach().cpu().numpy()
                    z_values = self.net(self.grid)[:, i].detach().cpu().numpy()

                    triang = Triangulation(grid_x, grid_y)

                    mask = (grid_x < 2) & (grid_y > 1)
                    triang.set_mask(mask[triang.triangles].any(axis=1))

                    ax1.plot_trisurf(triang, z_values,
                                     cmap=cm.jet, linewidth=0.2, alpha=1)
                    ax1.set_xlabel("x1")
                    ax1.set_ylabel("x2")

            elif self.img_dim == '2d':
                if nparams == 1:
                    ax1 = fig.add_subplot(1, self.nvars_model, i + 1)
                    if self.title is not None:
                        ax1.set_title(self.title + ' variable {}'.format(i))
                    ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                                self.net(self.grid)[:, i].detach().cpu().numpy())
                else:
                    ax1 = fig.add_subplot(1, self.nvars_model, i + 1)
                    if self.title is not None:
                        ax1.set_title(self.title + f' variable {i}')

                    grid_x = self.grid[:, 0].detach().cpu().numpy()
                    grid_y = self.grid[:, 1].detach().cpu().numpy()
                    z_values = self.net(self.grid)[:, i].detach().cpu().numpy()

                    axes_size = int(self.grid.shape[0] ** 0.5)

                    xi = np.linspace(grid_x.min(), grid_x.max(), axes_size)
                    yi = np.linspace(grid_y.min(), grid_y.max(), axes_size)
                    xi, yi = np.meshgrid(xi, yi)
                    zi = griddata((grid_x, grid_y), z_values, (xi, yi), method='cubic')

                    im = ax1.imshow(zi, extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
                                    origin='lower', cmap=cm.jet, aspect='auto')

                    ax1.set_xlabel("x1")
                    ax1.set_ylabel("x2")
                    fig.colorbar(im, ax=ax1)

            elif self.img_dim == '2d_scatter':
                if nparams == 1:
                    ax1 = fig.add_subplot(1, self.nvars_model, i + 1)
                    if self.title is not None:
                        ax1.set_title(self.title + ' variable {}'.format(i))
                    ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                                self.net(self.grid)[:, i].detach().cpu().numpy(),
                                marker='o', s=5, alpha=0.7)
                else:
                    ax1 = fig.add_subplot(1, self.nvars_model, i + 1)
                    if self.title is not None:
                        ax1.set_title(self.title + f' variable {i}')

                    grid_x = self.grid[:, 0].detach().cpu().numpy()
                    grid_y = self.grid[:, 1].detach().cpu().numpy()
                    z_values = self.net(self.grid)[:, i].detach().cpu().numpy()

                    scatter = ax1.scatter(
                        grid_x, grid_y,
                        c=z_values,
                        cmap=cm.jet,
                        s=5,
                        alpha=0.7
                    )

                    ax1.set_xlabel("x1")
                    ax1.set_ylabel("x2")
                    fig.colorbar(scatter, ax=ax1)

    def _plot_solution_2d(self, coord_3_values: List[float]):
        """
        Solution plot for 2D PDE u(x, y, t) at multiple time points.

        Args:
            t_values (List[float]): List of time points to plot the solution for.
        """

        grid_x = self.grid[:, 0]
        grid_y = self.grid[:, 1]

        self.nvars_model = self._init_nvars_model()

        if self.img_dim == '3d':
            for coord_3 in coord_3_values:
                coord_3_mask = self.grid[:, 2] == coord_3
                fig, axes = plt.subplots(1, self.nvars_model,
                                         figsize=(8 * self.nvars_model, 5 * self.nvars_model),
                                         subplot_kw={'projection': '3d'})

                if self.nvars_model == 1:
                    axes = [axes]

                for i in range(self.nvars_model):
                    z_values = self.net(self.grid[coord_3_mask])[:, i]
                    axes[i].plot_trisurf(grid_x[coord_3_mask].detach().cpu().numpy(),
                                         grid_y[coord_3_mask].detach().cpu().numpy(),
                                         z_values.detach().cpu().numpy().flatten(),
                                         cmap=cm.jet,
                                         linewidth=0.2,
                                         alpha=1)
                    axes[i].set_title(f'Function {i + 1} at t = {round(coord_3, 4)}')
                    axes[i].set_xlabel("x")
                    axes[i].set_ylabel("y")
                    axes[i].set_zlabel(f"func_{i + 1}(x, y)")

                plt.tight_layout()

                if self.img_dir:
                    directory = self._dir_path(self.img_dir, suffix=f"_2D_t{round(coord_3, 4)}")
                    plt.savefig(directory)
                plt.close(fig)

        elif self.img_dim == '2d':
            for coord_3 in coord_3_values:
                coord_3_mask = self.grid[:, 2] == coord_3
                fig, axes = plt.subplots(1, self.nvars_model,
                                         figsize=(6 * self.nvars_model, 5))

                if self.nvars_model == 1:
                    axes = [axes]

                for i in range(self.nvars_model):
                    z_values = self.net(self.grid[coord_3_mask])[:, i].detach().cpu().numpy().reshape(-1)

                    axes_size = int(self.grid.shape[0] ** 0.5)

                    xi = np.linspace(grid_x.detach().cpu().numpy().min(),
                                     grid_x.detach().cpu().numpy().max(),
                                     axes_size)
                    yi = np.linspace(grid_y.detach().cpu().numpy().min(),
                                     grid_y.detach().cpu().numpy().max(),
                                     axes_size)
                    zi = griddata((grid_x[coord_3_mask].detach().cpu().numpy(),
                                   grid_y[coord_3_mask].detach().cpu().numpy()),
                                  z_values,
                                  (xi[None, :], yi[:, None]),
                                  method='cubic')

                    im = axes[i].imshow(zi, extent=(grid_x.detach().cpu().numpy().min(),
                                                    grid_x.detach().cpu().numpy().max(),
                                                    grid_y.detach().cpu().numpy().min(),
                                                    grid_y.detach().cpu().numpy().max()),
                                        origin='lower', cmap=cm.jet, aspect='auto')

                    axes[i].set_title(f'Function {i + 1} at t = {coord_3}')
                    axes[i].set_xlabel("x")
                    axes[i].set_ylabel("y")
                    fig.colorbar(im, ax=axes[i])

                plt.tight_layout()

                if self.img_dir:
                    directory = self._dir_path(self.img_dir, suffix=f"_2D_t{coord_3}")
                    plt.savefig(directory)
                plt.close(fig)

        elif self.img_dim == '2d_scatter':
            for coord_3 in coord_3_values:
                coord_3_mask = self.grid[:, 2] == coord_3
                fig, axes = plt.subplots(1, self.nvars_model, figsize=(10 * self.nvars_model, 20))

                if self.nvars_model == 1:
                    axes = [axes]

                for i in range(self.nvars_model):
                    z_values = self.net(self.grid[coord_3_mask])[:, i].detach().cpu().numpy().flatten()
                    x_values = grid_x[coord_3_mask].detach().cpu().numpy()
                    y_values = grid_y[coord_3_mask].detach().cpu().numpy()

                    scatter = axes[i].scatter(
                        x_values,
                        y_values,
                        c=z_values,
                        cmap=cm.jet,
                        s=5,
                        alpha=0.7
                    )

                    axes[i].set_title(f'Function {i + 1} at t = {round(coord_3, 4)}')
                    axes[i].set_xlabel("x")
                    axes[i].set_ylabel("y")
                    fig.colorbar(scatter, ax=axes[i])

                plt.tight_layout()

                if self.img_dir:
                    directory = self._dir_path(self.img_dir, suffix=f"_2D_t{round(coord_3, 4)}")
                    plt.savefig(directory)
                plt.close(fig)

    def _plot_solution_3d(self, coord_4_values: List[float]):
        """
        Solution plot for 3D PDE u(x, y, z, t) at multiple time points.

        Args:
            t_values (List[float]): List of time points to plot the solution for.
        """
        grid_x = self.grid[:, 0].detach().cpu().numpy()
        grid_y = self.grid[:, 1].detach().cpu().numpy()
        grid_z = self.grid[:, 2].detach().cpu().numpy()

        for coord_4 in coord_4_values:
            coord_4_mask = self.grid[:, 3] == coord_4
            u_values = self.net(self.grid[coord_4_mask]).detach().cpu().numpy().flatten()

            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(grid_x[coord_4_mask], grid_y[coord_4_mask], grid_z[coord_4_mask],
                                 c=u_values, cmap=cm.jet)

            fig.colorbar(scatter, ax=ax, label="u(x, y, z)")

            ax.set_title(f'Solution of 3d equation u(x, y, z) at t = {round(coord_4, 4)}')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            if self.img_dim:
                directory = self._dir_path(self.img_dir, suffix=f"_3D_t{round(coord_4, 4)}")
                plt.savefig(directory)
            plt.close(fig)

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
                if self.grid.shape[1] == 2:
                    self._print_nn()

                    if save_flag:
                        directory = self._dir_path(self.img_dir)
                        plt.savefig(directory)
                    if print_flag:
                        plt.show()
                    plt.close()
                elif self.grid.shape[-1] == 3:
                    if self.t_values is None:
                        self.t_values = torch.unique(self.grid[:, 2]).detach().cpu().numpy()

                    selected_t_values = [self.t_values[0],
                                         self.t_values[len(self.t_values) // 4],
                                         self.t_values[len(self.t_values) // 2],
                                         self.t_values[len(self.t_values) // 4 * 3],
                                         self.t_values[-1]]
                    self._plot_solution_2d(selected_t_values)
                elif self.grid.shape[-1] == 4:
                    if self.t_values is None:
                        self.t_values = torch.unique(self.grid[:, 3]).detach().cpu().numpy()

                    selected_t_values = [self.t_values[0],
                                         self.t_values[len(self.t_values) // 4],
                                         self.t_values[len(self.t_values) // 2],
                                         self.t_values[len(self.t_values) // 4 * 3],
                                         self.t_values[-1]]
                    self._plot_solution_3d(selected_t_values)
                    self._plot_solution_2d(selected_t_values)

    def on_epoch_end(self, logs=None):
        self.solution_print()
