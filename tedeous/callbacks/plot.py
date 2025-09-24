import itertools
import os
import datetime
from typing import Union, List, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from scipy.interpolate import griddata
from tedeous.callbacks.callback import Callback


class Plots(Callback):
    """
    Class for plotting solutions.
    """


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
        Initializes the Plots class with parameters for controlling plot generation and display during the training process. These parameters allow for customization of the plots, such as frequency of display and saving, titles, and directory locations. This customization is essential for monitoring the training progress of neural networks used to solve differential equations, providing insights into the model's behavior and solution accuracy.
        
                Args:
                    print_every (Union[int, None], optional): Frequency (in steps) to display plots. Defaults to 500.
                    save_every (Union[int, None], optional): Frequency (in steps) to save plots. Defaults to 500.
                    title (str, optional): Title for the plots. Defaults to None.
                    img_dir (str, optional): Directory to save the plots. Defaults to None.
                    img_dim (str, optional): Dimensionality of the image data ('2d', '3d', '4d'). Defaults to None.
                    scatter_flag (bool): Whether to use scatter plots. Defaults to False.
                    plot_axes (List[int], optional): Axes to plot. Defaults to None.
                    fixed_axes (List[int], optional): Axes with fixed values. Defaults to None.
                    n_samples (int): Number of samples for fixed values. Defaults to 1.
                    img_rows (int, optional): Number of rows in the plot grid. Defaults to None.
                    img_cols (int, optional): Number of columns in the plot grid. Defaults to None.
                    var_transpose (bool): Whether to transpose variable axes. Defaults to False.
                    figsize (tuple): Figure size for the plots. Defaults to (15, 8).
        
                Returns:
                    None
        """
        super().__init__()
        self.print_every = print_every if print_every is not None else 0.1
        self.save_every = save_every if save_every is not None else 0.1
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
        self.nvars_model = None
        self.attributes = {'model': ['out_features', 'output_dim', 'width_out'],
                           'layers': ['out_features', 'output_dim', 'width_out']}

    def _init_nvars_model(self) -> int:
        """
        Determines the number of output variables of the neural network model used to approximate the solution of the differential equation.
        
        This function attempts to infer the number of output variables based on the model's architecture and attributes.
        It is crucial for correctly interpreting the model's output as a solution to the differential equation.
        
        Args:
            self: The Plots class instance.
        
        Returns:
            int: The number of output variables of the model, which corresponds to the dimensionality of the solution space.
        """
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
                nvars_model = self.net(self.grid).shape[-1] 
            except:
                nvars_model = self.net.width_out[-1]

        return nvars_model 

    def filter_grid(self, fixed_values: List[float]) -> np.ndarray:
        """
        Filters the solution grid to extract points corresponding to specific values on the fixed axes.
        
        This is useful for analyzing the solution at particular states or conditions
        defined by the fixed axes, allowing for a focused examination of the system's behavior.
        
        Args:
            fixed_values (List[float]): A list of values, one for each fixed axis,
                                        specifying the desired state.
        
        Returns:
            np.ndarray: A subset of the solution grid containing only the points
                        where the fixed axes match the provided values. If no fixed
                        axes are defined, the original grid is returned.
        """
        if self.fixed_axes is not None:
            for axis, value in zip(self.fixed_axes, fixed_values):
                mask = self.grid[:, axis] == value
            return self.grid[mask]
        else:
            return self.grid

    def generate_plot_data(self, subgrid: np.ndarray, nparams: int, i_ax: int, j_ax: int
                           ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generates the data required for visualizing the neural network's solution to the differential equation on a specified subgrid.
        
                This function extracts the predicted values and corresponding grid coordinates from the trained neural network,
                formatting them for plotting. It adapts to different model configurations, including single and multi-variable
                scenarios, to ensure accurate representation of the solution surface. This is a crucial step in understanding
                and validating how well the neural network approximates the true solution of the differential equation.
        
                Args:
                    subgrid (np.ndarray): A subset of the point grid where the solution is evaluated.
                    nparams (int): The number of parameters in the model, influencing how the output is processed.
                    i_ax (int): Index of the first axis, used for selecting the appropriate network output in multi-variable cases.
                    j_ax (int): Index of the second axis, also used for network output selection based on `var_transpose`.
        
                Returns:
                    Tuple[np.ndarray, List[np.ndarray]]: A tuple containing the function values (solution approximations)
                    and a list of grid coordinates corresponding to the plot axes.
        """
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

    def set_labels(self, i_ax: int, j_ax: int, ax: matplotlib.axes.Axes, nparams: int, fixed_values: list = None):
        """
        Configures the plot by setting axis labels and a title, which contextualize the visualization of the neural network's solution to the differential equation.
        
        Args:
            i_ax (int): the row index of the subgraph.
            j_ax (int): the column index of the subgraph.
            ax (matplotlib.axes.Axes): the axis on which the graph is plotted.
            nparams (int): the number of grid parameters (dimensionality of the problem).
            fixed_values (list, optional): values of fixed parameters, if any. Defaults to None.
        
        Returns:
            None: This method modifies the matplotlib axes object in place.
        
        Why:
            This method ensures that each subplot within the visualization is properly labeled and titled,
            providing clear context about the variables being plotted and any fixed parameter values.
            This is crucial for interpreting the neural network's solution to the differential equation.
        """
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

    def _plot_img(self, i_ax: int, j_ax: int, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, nparams: int,
                  fixed_values: list = None):
        """
        Generates and displays a solution plot on a subplot within a larger figure.
        
                This method takes preprocessed data and the trained model to visualize the solution
                of the differential equation. It supports 2D, 3D, and 4D plots, using scatter plots,
                imshow, or trisurf plots depending on the dimensionality and user preferences.
                Colorbars are added for higher-dimensional plots to represent the solution values.
        
                Args:
                    i_ax (int): The row index of the subplot in the figure's grid.
                    j_ax (int): The column index of the subplot in the figure's grid.
                    fig (matplotlib.figure.Figure): The matplotlib figure object.
                    ax (matplotlib.axes.Axes): The axes object for the current subplot.
                    nparams (int): The number of parameters in the input grid.
                    fixed_values (list, optional): A list of fixed values for certain parameters, used to create lower-dimensional slices of the solution space. Defaults to None.
        
                Returns:
                    None: The method modifies the axes object in place to display the solution plot.
        """
        subgrid = self.filter_grid(fixed_values)
        u_values, grid = self.generate_plot_data(subgrid, nparams, i_ax, j_ax)

        if self.img_dim == '2d':
            if nparams == 1:
                ax.scatter(grid[0].reshape(-1), u_values) if self.model.mode == 'mat' else \
                    ax.scatter(grid[0], u_values)
            else:
                if self.scatter_flag:
                    result_img = ax.scatter(*grid, c=u_values, cmap=cm.jet)
                else:
                    axes_size = int(self.grid.shape[0] ** 0.5)
                    xi, yi = np.meshgrid(*[np.linspace(grid_i.min(), grid_i.max(), axes_size) for grid_i in grid])
                    ui = griddata((grid[0], grid[1]), u_values, (xi, yi), method='cubic')
                    result_img = ax.imshow(ui, extent=(grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()),
                                           origin='lower', cmap=cm.jet, aspect='auto')
                if nparams > 1:
                    fig.colorbar(result_img, ax=ax)
        elif self.img_dim == '3d':
            if self.scatter_flag:
                ax.scatter(*grid, u_values, c=u_values, cmap=cm.jet)
            else:
                ax.plot_trisurf(*grid, u_values, cmap=cm.jet, linewidth=0.2, alpha=1)
        elif self.img_dim == '4d':
            result_img = ax.scatter(*grid, c=u_values.flatten(), cmap=cm.jet)
            fig.colorbar(result_img, ax=ax)

        self.set_labels(i_ax, j_ax, ax, nparams, fixed_values)

    def _dir_path(self, save_dir: str, suffix: str = "") -> str:
        """
        Generates a file path for saving a plot, ensuring the directory exists.
        
        This method determines the appropriate directory for saving a generated plot.
        If a `save_dir` is provided, it uses that directory. Otherwise, it defaults
        to a project-specific 'img' directory, creating it if it doesn't exist.
        The file name includes a timestamp and optional suffix to ensure uniqueness,
        facilitating the storage and organization of plots generated during the
        differential equation solving process. This ensures that each generated
        visualization can be saved and retrieved for analysis and comparison of
        different solution approaches.
        
        Args:
            save_dir (str, optional): The directory where the plot should be saved.
                If None, a default directory within the project is used.
            suffix (str, optional): A suffix to add to the filename. Defaults to "".
        
        Returns:
            str: The absolute path to the file where the plot will be saved.
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
        """
        Generates and displays or saves visualizations of the neural network's solution at specified intervals.
        
                This method creates plots of the solution obtained by the neural network, allowing for visual inspection of the solution's behavior over time or space. The plots can be either displayed on the screen or saved to a file, depending on the configuration. The method adapts the plot dimensions and axes based on the dimensionality of the problem and the number of variables being visualized.
        
                Args:
                    self: The object instance containing the model, grid, and plotting configurations.
        
                Returns:
                    None. The method generates plots as a side effect.
        
                WHY: This method is essential for understanding and validating the neural network's solution. By visualizing the solution, users can assess its accuracy, identify potential issues, and gain insights into the underlying dynamics of the differential equation.
        """

        print_flag = self.model.t % self.print_every == 0
        save_flag = self.model.t % self.save_every == 0

        if print_flag or save_flag:
            self.net = self.model.net
            self.grid = self.model.solution_cls.grid
            self.nvars_model = self._init_nvars_model()
            nparams = self.grid.shape[0] if self.model.mode == 'mat' else self.grid.shape[1]

            self.img_dim = self.img_dim or (f"{nparams + 1}d" if nparams + 1 < 5 else '2d')
            self.img_rows = self.img_rows or (self.nvars_model if nparams > 2 else 1)
            self.img_cols = self.img_cols or (self.n_samples if nparams > 2 else self.nvars_model)
            self.plot_axes = self.plot_axes or (list(range(nparams)) if nparams < 3 else list(range(nparams))[:2])
            self.fixed_axes = self.fixed_axes or (None if nparams < 3 else list(range(nparams))[2:])

            fixed_points_combinations = (
                range(self.img_cols) if self.fixed_axes is None else
                list(itertools.product(*[
                    torch.unique(self.grid[:, axis]).detach().cpu().numpy()[
                        torch.linspace(0, len(torch.unique(self.grid[:, axis])) - 1,
                                       self.n_samples).long().detach().cpu().numpy()
                    ] for axis in self.fixed_axes
                ]))
            )

            n_rows, n_cols = (self.img_cols, self.img_rows) if self.var_transpose else (self.img_rows, self.img_cols)
            subplot_kwargs = {'subplot_kw': {'projection': '3d'}} if self.img_dim in ('3d', '4d') else {}

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(self.figsize[0] * n_cols, self.figsize[1] * n_rows),
                squeeze=False,
                **subplot_kwargs
            )

            i_fix_value = 0 if nparams > 2 else None

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
                    self._plot_img(i_ax, j_ax, fig, ax, nparams, fixed_values)

                if nparams > 2 and self.nvars_model > 1 and self.var_transpose:
                    i_fix_value += 1

            if save_flag:
                plt.savefig(self._dir_path(self.img_dir))
            if print_flag:
                plt.show()

            plt.close()

    def on_epoch_end(self, logs=None):
        """
        Logs the state of the solution at the end of each epoch.
        
        This is done to track the progress of the neural network's approximation
        of the differential equation's solution during training. By printing
        the solution, we can observe how the network's output evolves over time
        and assess its convergence towards the true solution.
        
        Args:
            logs (dict, optional): The logs returned by the Keras model. Defaults to None.
        
        Returns:
            None
        """
        self.solution_print()
