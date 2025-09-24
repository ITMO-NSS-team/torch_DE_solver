# This code is partially based on the repository source: https://github.com/elhamod/NeuroVisualizer.git.
import csv
import json
import pandas as pd
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, NoNorm
from typing import List
from collections import OrderedDict

from landscape_visualization._aux.AEmodel import UniformAutoencoder
from landscape_visualization._aux.trajectories_data import get_trajectory_dataloader
from landscape_visualization._aux.utils import get_density, get_files, repopulate_model
from landscape_visualization._aux.PINN_loss_data import PINNLossData, get_PINN

import tedeous.model as model
from tedeous.data import Domain, Conditions, Equation


class PlotLossSurface:
    """
    Class for preprocessing plot loss surface
    """


    def __init__(self,
                 path_to_plot_model: str = None,
                 path_to_trajectories: str = None,
                 solver_models: List[torch.nn.Module] = None,
                 AE_model: torch.nn.Module = None,
                 key_models: list = None,
                 key_modelnames: list = None,
                 prefix: str = "model-",
                 num_models: int = None,
                 from_last: bool = True,
                 vlevel: int = 20,
                 vmin: float = -1,
                 vmax: float = 1,
                 x_range: list = [-1.2, 1.2, 25],
                 loss_types: list = ["loss_total", "loss_oper", "loss_bnd"],
                 loss_type: str = "loss_total",
                 loss_name: str = "train_loss",
                 layers_AE: list = [991, 125, 15],
                 num_of_layers: int = 3,
                 batch_size: int = 32,
                 every_nth: int = 1,
                 density_type: str = "CKA",
                 density_p: float = 2,
                 density_vmax: float = -1,
                 density_vmin: float = -1,
                 colorFromGridOnly: bool = True,
                 img_dir: str = None
                 ):

        """
        Initializes the PlotLossSurface class.
        
        This class is designed to visualize the loss landscape and trajectory of neural network models
        trained to solve differential equations. It sets up the necessary parameters and configurations
        for plotting, including loading model trajectories, defining the grid for loss surface evaluation,
        and preparing for visualization. The initialization process involves setting up paths, loading models,
        and defining the range and resolution of the plot.
        
        Args:
            path_to_plot_model (str): Path to the saved model file used for plotting.
            path_to_trajectories (str): Path to the directory containing models trajectory.
            solver_models (List[torch.nn.Module]): weights of solver model. Defaults to None.
            AE_model (torch.nn.Module): weights of autoencoder model. Defaults to None.
            key_models (list, optional): List of indices of key models to highlight during plotting. Defaults to None.
            key_modelnames (list, optional): List of names corresponding to the key models. Defaults to None.
            prefix (str, optional): Prefix used to identify model files in the directory. Defaults to "model-".
            num_models (int, optional): Number of models to consider in the trajectories. Defaults to None.
            from_last (bool, optional): Whether to consider models starting from the last file in the directory. Defaults to True.
            vlevel (int, optional): Number of contour levels for the loss surface plot. Defaults to 20.
            vmin (float, optional): Minimum value for the loss surface plot. Defaults to -1.
            vmax (float, optional): Maximum value for the loss surface plot. Defaults to 1.
            x_range (list, optional): Range of x-coordinates for the loss surface grid in the format [min_x, max_x, num_points]. Defaults to [-1.2, 1.2, 25].
            loss_types (list, optional): List of loss types to evaluate, may be "loss_total", "loss_oper", "loss_bnd". Defaults to ["loss_total", "loss_oper", "loss_bnd"].
            loss_type (str, optional): Type of loss to evaluate, may be "loss_total", "loss_oper", "loss_bnd". Defaults to "loss_total".
            loss_name (str, optional): Name of the loss to be plotted. Defaults to "train_loss".
            layers_AE (list, optional): Structure of the layers in the autoencoder model. Defaults to [991, 125, 15].
            num_of_layers (int, optional): Number of layers in the autoencoder model. Defaults to 3.
            batch_size (int, optional): Batch size used during data loading. Defaults to 32.
            every_nth (int, optional): Only consider every nth model in the trajectory data. Defaults to 1.
            density_type (str, optional): Type of density function to use (e.g., "CKA"). Defaults to "CKA".
            density_p (float, optional): Parameter for density function calculation. Defaults to 2.
            density_vmax (float, optional): Maximum density value for visualization. Defaults to -1.
            density_vmin (float, optional): Minimum density value for visualization. Defaults to -1.
            colorFromGridOnly (bool, optional): Whether to derive color limits only from the grid data. Defaults to True.
            img_dir (str, optional): directory title where plots are being saved. Defaults to None.
        
        Returns:
            None
        """

        self.path_to_plot_model = path_to_plot_model
        self.path_to_trajectories = path_to_trajectories
        self.solver_models = solver_models
        self.AE_model = AE_model
        self.key_models = key_models
        self.key_modelnames = key_modelnames
        self.prefix = prefix
        self.num_models = num_models
        self.from_last = from_last
        self.vlevel = vlevel
        self.vmin = vmin
        self.vmax = vmax
        self.x_range = x_range
        self.loss_type = loss_type
        self.loss_name = loss_name
        self.layers_AE = layers_AE
        self.num_of_layers = num_of_layers
        self.batch_size = batch_size
        self.every_nth = every_nth
        self.density_type = density_type
        self.density_p = density_p
        self.density_vmax = density_vmax
        self.density_vmin = density_vmin
        self.colorFromGridOnly = colorFromGridOnly
        self.loss_types = loss_types
        self.latent_dim = 2
        self.img_dir = img_dir
        self.states_dict = {}
        self.counter = 1

        if self.path_to_plot_model:
            self.path_to_plot_model_directory = os.path.dirname(self.path_to_plot_model)
            if not os.path.exists(self.path_to_plot_model_directory):
                os.makedirs(self.path_to_plot_model_directory)
            # Convert args to JSON format
            args_dict = vars(self)  # Convert Namespace object to dictionary
            json_str = json.dumps(args_dict, indent=4)  # Convert dictionary to JSON string
            # Save JSON to file
            with open(os.path.join(self.path_to_plot_model_directory, 'plotting_args.json'), 'w') as f:
                f.write(json_str)

        self.min_x, self.max_x, self.xnum = self.x_range
        self.step_size = (self.max_x - self.min_x) / self.xnum
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.get_trajectories_and_load_model()

    def get_errors(self, model: torch.nn.Module, error_type: str, loss_dict: dict):
        """
        Computes a specified error metric to evaluate the model's performance in solving the differential equation. This function is used to quantify the difference between the model's approximation and the true solution or loss components.
        
                Args:
                    model (torch.nn.Module): The neural network model used to approximate the solution of the differential equation.
                    error_type (str): Specifies the type of error to compute. Options include:
                        - "u": Root mean squared error between the exact solution (`u_exact_test`) and the model's prediction on the test grid.  This evaluates the model's accuracy in approximating the solution.
                        - "loss_oper": The operational loss component extracted from the loss dictionary. This reflects how well the model satisfies the differential equation within the domain.
                        - "loss_bnd": The boundary loss component extracted from the loss dictionary. This reflects how well the model satisfies the boundary conditions.
                        - "loss_total": The total loss extracted from the loss dictionary. This represents the overall error in the model's approximation.
                    loss_dict (dict): A dictionary containing loss components such as "loss_oper", "loss_bnd", and "loss". These losses are computed during the training process.
        
                Returns:
                    torch.Tensor: The computed error value based on the specified `error_type`.  This value provides a quantitative measure of the model's performance.
        """
        if error_type == "u":
            error = torch.sqrt(torch.mean((self.u_exact_test - model(self.grid_test).reshape(-1)) ** 2))
        if error_type == "loss_oper":
            error = loss_dict["loss_oper"].detach().cpu()
        if error_type == "loss_bnd":
            error = loss_dict["loss_bnd"].detach().cpu()
        if error_type == "loss_total":
            error = loss_dict["loss"].detach().cpu()
        return error

    def get_trajectories_and_load_model(self):
        """
        Loads trajectory data and a pre-trained autoencoder model.
        
        This method prepares the necessary components for visualizing the loss surface. It either loads trajectory data from `.pt` files or uses provided solver models to create a dataset. It also loads a pre-trained autoencoder model, which is later used to map the high-dimensional trajectory data into a lower-dimensional latent space for visualization. This is a crucial step in understanding the optimization landscape and the behavior of the trained models.
        
        Args:
            self: The PlotLossSurface instance.
        
        Returns:
            None. Sets the following attributes:
                - best_model: The loaded autoencoder model.
                - transform: The data transformation used for the trajectory data.
                - trajectory_dataset: The loaded trajectory dataset.
        """
        if self.solver_models:
            solver_models_state_dicts = [solver_model.state_dict() for solver_model in self.solver_models]
            trajectory_data_loader, transform = get_trajectory_dataloader(
                self.batch_size, models=solver_models_state_dicts
            )
            trajectory_dataset = trajectory_data_loader.dataset
            best_model = self.AE_model
        else:
            pt_files = get_files(self.path_to_trajectories, self.num_models, prefix=self.prefix,
                                 from_last=self.from_last,
                                 every_nth=self.every_nth)

            trajectory_data_loader, transform = get_trajectory_dataloader(
                self.batch_size, pt_files=pt_files
            )
            trajectory_dataset = trajectory_data_loader.dataset
            input_dim = trajectory_dataset[0].shape[0]

            best_model = UniformAutoencoder(input_dim, self.num_of_layers, self.latent_dim, h=self.layers_AE).to(
                self.device)
            best_model.load_state_dict(torch.load(self.path_to_plot_model, map_location=torch.device('cpu')))

        self.best_model = best_model.to(self.device)
        self.best_model.eval()

        self.transform = transform
        self.trajectory_dataset = trajectory_dataset

    def compute_losses(self, models, domain, equation, boundaries, PINN_layers):
        """
        Computes the loss values for a set of neural network models, enabling the analysis of their performance in solving the differential equation. This is crucial for understanding how different model configurations affect the accuracy and stability of the solution.
        
                Args:
                    models (torch.Tensor): A tensor containing flattened model parameters. Each row represents a different model.
                    domain (Domain): The domain over which the differential equation is defined.
                    equation (Equation): The differential equation to be solved.
                    boundaries (Boundaries): The boundary conditions for the differential equation.
                    PINN_layers (list): A list defining the architecture of the Physics-Informed Neural Network (PINN) layers.
        
                Returns:
                    dict: A dictionary where keys are loss types (e.g., 'equation', 'boundary') and values are tensors containing the corresponding loss values for each model.
        """
        losses_dict = {}
        for loss_type in self.loss_types:
            losses_dict[loss_type] = []
        for i in range(models.shape[0]):
            model_flattened = models[i, :]
            model_repopulated = repopulate_model(model_flattened, get_PINN(PINN_layers, self.device))
            model_repopulated = model_repopulated.to(self.device)

            equation_model = model.Model(model_repopulated, domain, equation, boundaries)
            equation_model.compile('autograd', lambda_operator=1, lambda_bound=100)

            loss_compute = PINNLossData(equation_model.solution_cls)
            loss_dict = loss_compute.evaluate(save_graph=False)
            for loss_type in self.loss_types:
                losses_dict[loss_type].append(loss_dict[loss_type])

        for loss_type in self.loss_types:
                losses_dict[loss_type] = torch.stack(losses_dict[loss_type])
        
        return losses_dict 

    def get_coordinates_and_losses_of_trajectories(self, grid, domain, equation, boundaries, PINN_layers):
        """
        Evaluates trajectory models and original samples to quantify their equation-solving performance.
        
        This method leverages a trained autoencoder to map high-dimensional model parameters
        into a lower-dimensional latent space, and then assesses the equation-solving
        accuracy of both the encoded (trajectory) models and the original samples. This helps
        to understand how well the autoencoder preserves the equation-solving properties
        of the original models in the reduced latent space.
        
        Args:
            grid (torch.Tensor): Discretization of the computational domain.
            domain (Domain): Object representing the problem domain.
            equation (Equation): Object representing the differential equation.
            boundaries (Conditions): Object representing the boundary conditions.
            PINN_layers (list): List of layers used for repopulating models.
        
        Returns:
            tuple: A tuple containing:
                - trajectory_losses (torch.Tensor): Losses computed for each trajectory autoencoder model,
                  indicating how well the encoded models satisfy the equation.
                - original_trajectory_losses (torch.Tensor): Losses for the original trajectory models,
                  serving as a baseline for comparison.
                - trajectory_coordinates (torch.Tensor): Latent-space coordinates for the trajectory models,
                  representing the encoded model parameters.
        """

        print("Get coordinates and losses of trajectories")
        trajectory_coordinates = []
        trajectory_dataset_samples = []
        trajectory_coordinates_rec = []

        with torch.no_grad():
            for data in self.trajectory_dataset:
                data = data.to(self.device).view(1, -1).float()
                x_recon, z = self.best_model(data)

                trajectory_coordinates.append(z)
                trajectory_coordinates_rec.append(x_recon)
                trajectory_dataset_samples.append(data)

        trajectory_coordinates = torch.cat(trajectory_coordinates, dim=0).cpu()
        trajectory_models = torch.cat(trajectory_coordinates_rec, dim=0).cpu()
        original_models = torch.cat(trajectory_dataset_samples, dim=0).cpu()

        # Денормализация данных
        trajectory_models = trajectory_models * self.transform.std.cpu() + self.transform.mean.cpu()
        original_models = original_models * self.transform.std.cpu() + self.transform.mean.cpu()

        # Вычисление ошибки для моделей в траектории
        trajectory_losses = self.compute_losses(trajectory_models, domain, equation, boundaries, PINN_layers)
        original_trajectory_losses = self.compute_losses(original_models, domain, equation, boundaries, PINN_layers)

        return trajectory_losses, original_trajectory_losses, trajectory_coordinates

    def get_coordinates_and_losses_of_surface(self, grid, domain, equation, boundaries, PINN_layers):
        """
        Generates coordinates and evaluates losses on the loss surface.
        
        This method samples points on a predefined grid, reconstructs corresponding model parameters using the decoder,
        and then calculates the loss values for these reconstructed models. This process allows for visualization
        and analysis of the loss landscape around the optimal solution.
        
        Args:
            grid (torch.Tensor): Discretization of the latent space domain.
            domain (Domain): Object representing the computational domain.
            equation (Equation): Object representing the differential equation.
            boundaries (Conditions): Object representing the boundary conditions.
            PINN_layers (list): List of layers used for repopulating models.
        
        Returns:
            tuple: A tuple containing:
                - grid_losses (dict): A dictionary of computed losses for each point in the loss surface grid.
                - grid_xx (torch.Tensor): A 2D tensor representing the x-coordinates of the grid.
                - grid_yy (torch.Tensor): A 2D tensor representing the y-coordinates of the grid.
                - rec_grid_models (torch.Tensor): A tensor of reconstructed models obtained by decoding the grid points.
        """
        print("Get coordinates and losses of surface")
        # scan the unit plane from 0-1 for 2D.
        # For each step, evalute the coordinate through the decoder and get the parameters and then get the loss.

        self.min_x, self.max_x, self.xnum = self.x_range

        min_x, max_x = self.min_x, self.max_x
        min_y, max_y = self.min_x, self.max_x

        x_coords = torch.arange(min_x, max_x + self.step_size, self.step_size)
        y_coords = torch.arange(min_y, max_y + self.step_size, self.step_size)

        grid_xx, grid_yy = torch.meshgrid(x_coords, y_coords)
        grid_coords = torch.stack((grid_xx.flatten(), grid_yy.flatten()), dim=1).to(self.device)

        rec_grid_models = self.best_model.decoder(grid_coords)
        rec_grid_models = rec_grid_models * self.transform.std.to(self.device) + self.transform.mean.to(self.device)

        grid_losses = self.compute_losses(rec_grid_models, domain, equation, boundaries, PINN_layers)
        for loss_type in self.loss_types:
                grid_losses[loss_type] = grid_losses[loss_type].view(grid_xx.shape)

        return grid_losses, grid_xx, grid_yy, rec_grid_models

    def plotting(self, trajectory_losses: torch.Tensor, original_trajectory_losses: torch.Tensor,
                 trajectory_coordinates: torch.Tensor, grid_losses: torch.Tensor,
                 grid_xx: torch.Tensor, grid_yy: torch.Tensor, rec_grid_models: torch.Tensor):
        """
        Plots the loss surface and trajectory of solutions in the latent space.
        
                This method visualizes the loss landscape by plotting the loss values over a grid
                of points in the latent space, along with the trajectory of the autoencoder models.
                It helps to understand the behavior of the models during training and the characteristics
                of the loss surface. The method also generates a density map of the reconstructed models
                on the grid.
        
                Args:
                    trajectory_losses (torch.Tensor):  A tensor of losses computed for each trajectory autoencoder model.
                    original_trajectory_losses (torch.Tensor): Tensor of the original losses.
                    trajectory_coordinates (torch.Tensor): Tensor of latent-space coordinates for the trajectory models.
                    grid_losses (torch.Tensor): Tensor of losses for the loss surface grid.
                    grid_xx (torch.Tensor): 2D tensor representing the x-coordinates of the grid for the loss surface.
                    grid_yy (torch.Tensor): 2D tensor representing the y-coordinates of the grid for the loss surface.
                    rec_grid_models (torch.Tensor): Tensor containing the reconstructed models obtained by decoding grid points in the latent space.
        
                Returns:
                    None
        """
        vmax = self.vmax
        vmin = self.vmin
        if self.vmax <= 0 or self.vmin <= 0:
            if self.colorFromGridOnly == False:
                if self.vmax <= 0:
                    vmax = max(torch.max(grid_losses).detach().cpu().numpy(),
                               torch.max(original_trajectory_losses).detach().cpu().numpy())
                    vmax = vmax * 1.1
                if self.vmin <= 0:
                    vmin = min(torch.min(grid_losses).detach().cpu().numpy(),
                               torch.min(original_trajectory_losses).detach().cpu().numpy())
                    vmin = vmin / 1.1
            else:
                if self.vmax <= 0:
                    vmax = torch.max(grid_losses).detach().cpu().numpy() * 1.1
                if self.vmin <= 0:
                    vmin = torch.min(grid_losses).detach().cpu().numpy() / 1.1

        print(f"Auto calculated: [vmin, vmax] = [{vmin}, {vmax}]")
        print('Plotting')

        levels = np.logspace(np.log10(vmin), np.log10(vmax), int(self.vlevel))

        plots_ = ['loss', 'relative_error', 'abs_error', 'dists_param_space']
        df = pd.DataFrame(columns=['index', 'file', 'x', 'y'] + plots_)

        relative_errors = (torch.abs(trajectory_losses - original_trajectory_losses) / original_trajectory_losses)
        abs_errors = (torch.abs(trajectory_losses - original_trajectory_losses))
        ds = []
        for batch_idx, data in enumerate(self.trajectory_dataset):
            data = data.to(self.device).float()

            x_recon, z = self.best_model(data.view(1, -1))
            z = z.view(-1)

            transform = self.trajectory_dataset.transform
            data_unnormalized = data * transform.std.to(self.device) + transform.mean.to(self.device)
            x_recon_unnormalized = x_recon * transform.std.to(self.device) + transform.mean.to(self.device)
            d = (data_unnormalized - x_recon_unnormalized).pow(2).sum().sqrt()
            ds.append(d)

            row = {
                'index': batch_idx,
                # 'file': os.path.basename(self.trajectory_dataset.current_weighs[batch_idx]),
                'x': z[0].detach().cpu().numpy(),
                'y': z[1].detach().cpu().numpy(),
                'dists_param_space': d.item(),
                'loss': trajectory_losses[batch_idx].item(),
                'original_loss': original_trajectory_losses[batch_idx].item(),
                'relative_error': relative_errors[batch_idx].item(),
                'abs_error': abs_errors[batch_idx].item()}

            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # Calculate the mean of the specified columns
        mean_row = pd.DataFrame(df[['abs_error', 'relative_error', 'dists_param_space']].mean(axis=0)).T
        # Add a column to the mean row with the string 'Mean'
        mean_row['index'] = 'Mean'
        # Append the mean row to the original DataFrame
        df = pd.concat([df, mean_row], ignore_index=True)

        if self.solver_models:
            df.to_csv(os.path.join(self.img_dir,
                                   'summary_' + self.loss_name + '_' + self.loss_type + f'_opt_{self.counter}' + '.csv'),
                      quoting=csv.QUOTE_NONNUMERIC, index=False)
        else:
            df.to_csv(os.path.join(self.path_to_plot_model_directory,
                                   'summary_' + self.loss_name + '_' + self.loss_type + '.csv'),
                      quoting=csv.QUOTE_NONNUMERIC, index=False)

        ds = torch.stack(ds)

        name_map = {
            'train_loss': 'Training',
            'test_loss': 'Test',
            'val_loss': 'Validation',
            'loss': 'loss value',
            'relative_error': 'relative loss error',
            'abs_error': 'absolute loss error',
            'dists_param_space': 'projection error in parameter space',
        }

        fig = plt.figure()
        ax = plt.gca()
        norm = NoNorm()
        density = get_density(rec_grid_models.detach().cpu().numpy(), self.density_type, self.density_p)
        if self.density_vmax <= 0 or self.density_vmin <= 0:
            density_vmax = np.max(density)
            density_vmax = density_vmax * 1.1
            density_vmin = np.min(density)
            density_vmin = density_vmin / 1.1
            print(f"Auto calculated: [density_vmin, density_vmax] = [{density_vmin}, {density_vmax}]")
        else:
            density_vmax = self.density_vmax
            density_vmin = self.density_vmin
            print(f"[density_vmin, density_vmax] = [{density_vmin}, {density_vmax}]")

        levels_density = np.linspace(density_vmin, density_vmax, int(self.vlevel))
        density = density.reshape(list(grid_xx.shape))
        CS = plt.contour(grid_xx.detach().cpu().numpy(), grid_yy.detach().cpu().numpy(), density, levels=levels_density,
                         vmin=density_vmin, vmax=density_vmax)
        fmt = ticker.FormatStrFormatter('%.2e')
        cbar = plt.colorbar(CS)
        scatter = plt.scatter(trajectory_coordinates[:, 0].detach().cpu().numpy(),
                              trajectory_coordinates[:, 1].detach().cpu().numpy(), c='0.5', marker='o', s=9, zorder=100)
        cbar.ax.set_ylabel("Density")

        if self.solver_models:
            fig.savefig(os.path.join(self.img_dir, 'map_' + self.loss_type + f'_opt_{self.counter}'
                                     + '_grid_density.pdf'),
                        dpi=300, bbox_inches='tight', format='pdf')
        else:
            fig.savefig(os.path.join(self.path_to_plot_model_directory, 'map_' + self.loss_type + '_grid_density.pdf'),
                        dpi=300, bbox_inches='tight', format='pdf')

        fig.show()

        min_x, max_x = self.min_x, self.max_x
        min_y, max_y = self.min_x, self.max_x
        for plot_ in plots_:
            fig = plt.figure()

            ax = plt.gca()
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])

            norm = LogNorm()

            CS = plt.contour(grid_xx.detach().cpu().numpy(), grid_yy.detach().cpu().numpy(),
                             grid_losses.detach().cpu().numpy(), levels=levels, norm=norm)

            fmt = ticker.FormatStrFormatter('%.2e')
            ax.clabel(CS, CS.levels, fmt=lambda x: fmt(x), inline=1, fontsize=7)

            cmap = None

            if plot_ == 'relative_error':
                c = relative_errors.detach().cpu().numpy()
            elif plot_ == 'abs_error':
                c = abs_errors.detach().cpu().numpy()
            elif plot_ == 'loss':
                c = original_trajectory_losses.detach().cpu().numpy()
                cmap = CS.cmap
            elif plot_ == 'dists_param_space':
                c = ds.detach().cpu().numpy()
            else:
                raise "Unknown polt type"

            scatter = plt.scatter(trajectory_coordinates[:, 0].detach().cpu().numpy(),
                                  trajectory_coordinates[:, 1].detach().cpu().numpy(), c=c, marker='o', s=9, norm=norm,
                                  cmap=cmap, zorder=100)  # , edgecolors='k'

            if self.key_models is not None:
                for i, idx in enumerate(self.key_models):
                    key_model_indx = int(idx)
                    key_modelname = self.key_modelnames[i]
                    plt.scatter(trajectory_coordinates[:, 0][key_model_indx].detach().cpu().numpy(),
                                trajectory_coordinates[:, 1][key_model_indx].detach().cpu().numpy(), c=c[0], marker='o',
                                s=8, norm=norm, edgecolors='k', cmap=cmap, zorder=100, linewidths=2)

                    if self.key_modelnames is not None:
                        if i == len(self.key_models) - 1:
                            last_key_model_indx = trajectory_coordinates.shape[0] - 1
                        else:
                            last_key_model_indx = int(self.key_models[i + 1]) - 1
                        plt.text(trajectory_coordinates[:, 0][last_key_model_indx].detach().cpu().numpy(),
                                 trajectory_coordinates[:, 1][last_key_model_indx].detach().cpu().numpy(),
                                 key_modelname, ha='left', va='top', zorder=101, fontsize=9,
                                 backgroundcolor=(1.0, 1.0, 1.0, 0.5))

            # Connect the dots with lines
            if self.key_models is None:
                x = trajectory_coordinates[:, 0].detach().cpu().numpy()
                y = trajectory_coordinates[:, 1].detach().cpu().numpy()
                for i in range(len(x) - 1):
                    plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color='k')
            else:
                n = 0
                for j, idx in enumerate(self.key_models):
                    if j == len(self.key_models) - 1:
                        key_model_indx = trajectory_coordinates.shape[0]
                    else:
                        key_model_indx = int(self.key_models[j + 1])

                    x = trajectory_coordinates[:, 0][n:key_model_indx].detach().cpu().numpy()
                    y = trajectory_coordinates[:, 1][n:key_model_indx].detach().cpu().numpy()
                    for i in range(len(x) - 1):
                        plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color='k')
                    n = key_model_indx

            cbar = plt.colorbar(scatter, shrink=0.6)
            cbar.ax.set_ylabel(name_map[plot_])

            if self.solver_models:
                fig.savefig(os.path.join(self.img_dir,
                                         'map_' + self.loss_type + '_' + self.loss_name + '_' + plot_ +
                                         f'_opt_{self.counter}' + '.pdf'), dpi=300,
                            bbox_inches='tight', format='pdf')
            else:
                fig.savefig(os.path.join(self.path_to_plot_model_directory,
                                         'map_' + self.loss_type + '_' + self.loss_name + '_' + plot_ + '.pdf'),
                            dpi=300,
                            bbox_inches='tight', format='pdf')

            fig.show()

    def plotting_equation_loss_surface(self, u_exact_test: torch.Tensor, grid_test: torch.Tensor, grid: torch.Tensor,
                                       domain: Domain, equation: Equation, boundaries: Conditions, PINN_layers: list):

        """
        Preprocesses and orchestrates the plotting of loss surfaces and training trajectories for neural network-based differential equation solvers. It prepares the data by computing losses over parameter space and along optimization paths, then visualizes these losses to analyze the training process and solution landscape. This allows for insights into how network parameters affect solution accuracy and training stability.
        
                Args:
                    u_exact_test (torch.Tensor): The exact solution of the equation used for computing test errors.
                    grid_test (torch.Tensor): The test grid on which the exact solution and predictions are compared.
                    grid (torch.Tensor): Discretization of the computational domain.
                    domain (Domain): Object of class Domain, defining the problem domain.
                    equation (Equation): Object of class Equation, defining the differential equation.
                    boundaries (Conditions): Object of class Conditions, specifying boundary conditions.
                    PINN_layers (list): List of layers used for repopulating models.
        
                Returns:
                    None: This method orchestrates plotting and does not return any value. The plotting is done internally using the processed data.
        """
        self.grid_test = grid_test
        self.u_exact_test = u_exact_test

        trajectory_losses, original_trajectory_losses, trajectory_coordinates = \
            self.get_coordinates_and_losses_of_trajectories(grid, domain, equation, boundaries, PINN_layers)

        grid_losses, grid_xx, grid_yy, rec_grid_models = self.get_coordinates_and_losses_of_surface(
            grid, domain, equation, boundaries, PINN_layers)
        
        for loss_type in self.loss_types: 
            self.loss_type = loss_type
            self.plotting(trajectory_losses[loss_type], original_trajectory_losses[loss_type], trajectory_coordinates,
                      grid_losses[loss_type], grid_xx, grid_yy, rec_grid_models)

    def save_equation_loss_surface(self, u_exact_test: torch.Tensor, grid_test: torch.Tensor, grid: torch.Tensor,
                                   domain: Domain, equation: Equation, boundaries: Conditions, PINN_layers: list):
        """
        Generates and saves data for visualizing the loss surface of the equation component of the PINN.
        
        This method computes the loss landscape and trajectory data, storing it for later visualization.
        This allows to analyze the optimization process and the characteristics of the loss function.
        
        Args:
            u_exact_test (torch.Tensor): Exact solution on the test grid.
            grid_test (torch.Tensor): Test grid for evaluating the exact solution.
            grid (torch.Tensor): Discretization of the computational domain.
            domain (Domain): Object representing the domain of the equation.
            equation (Equation): Object representing the differential equation.
            boundaries (Conditions): Object representing the boundary conditions.
            PINN_layers (list): List of layers used for repopulating models.
        
        Returns:
            dict: A dictionary containing the loss values on the grid for each loss type.
                  This data is essential for plotting the loss surface.
        """
        raw_states_dict = {}
        self.grid_test = grid_test
        self.u_exact_test = u_exact_test

        trajectory_losses, original_trajectory_losses, trajectory_coordinates = \
            self.get_coordinates_and_losses_of_trajectories(grid, domain, equation, boundaries, PINN_layers)

        grid_losses, grid_xx, grid_yy, rec_grid_models = \
            self.get_coordinates_and_losses_of_surface(grid, domain, equation, boundaries, PINN_layers)
        
        for loss_type in self.loss_types:
            raw_state = {
                'grid_losses': grid_losses[loss_type],
                'grid_xx': grid_xx,
                'grid_yy': grid_yy,
                'trajectory_losses': trajectory_losses[loss_type],
                'original_trajectory_losses': original_trajectory_losses[loss_type],
                'trajectory_coordinates': trajectory_coordinates
            }

            self.states_dict[loss_type] = raw_state

            if self.solver_models is None:
                torch.save(self.states_dict, self.path_to_plot_model_directory + '/loss_surface_data.pt')
            
            raw_states_dict[loss_type] = raw_state['grid_losses']

        return raw_states_dict
