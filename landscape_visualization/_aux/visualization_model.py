# This code is partially based on the repository source: https://github.com/elhamod/NeuroVisualizer.git.

import json
import pandas as pd
import torch
import os
import itertools
from typing import Union, List, Any

from tedeous.callbacks.callback_list import CallbackList
from tedeous.optimizers.optimizer import Optimizer
import warnings
warnings.filterwarnings("ignore", message="The frame.append method is deprecated")


from landscape_visualization._aux.AEmodel import UniformAutoencoder
from landscape_visualization._aux.losses_of_plot import loss_grid_to_trajectory, rec_loss_function, loss_anchor
from landscape_visualization._aux.trajectories_data import get_trajectory_dataloader, get_anchor_dataloader, get_predefined_values
from landscape_visualization._aux.utils import get_files, get_gridpoint_and_trajectory_datasets, loss_well_spaced_trajectory, plot_losses


class VisualizationModel:
    """Class for preprocessing"""
    def __init__(self, 
                 mode: str,
                 num_of_layers: int, 
                 layers_AE: list, 
                 path_to_plot_model: str, 
                 path_to_trajectories: str,
                 num_models: int = None, 
                 from_last: bool = False, 
                 prefix: str = 'model-', 
                 every_nth: int = 1, 
                 grid_step: float = 0.1, 
                 d_max_latent: float = 2.0, 
                 anchor_mode: str = "circle", 
                 rec_weight: float = 1.0, 
                 anchor_weight: float = 0.0, 
                 lastzero_weight: float = 0.0, 
                 polars_weight: float = 0.0, 
                 wellspacedtrajectory_weight: float=0.0, 
                 gridscaling_weight: float = 0.0, 
                #  resume: bool = False
                 ):
        
        """
        Args:
            mode (str): The training mode for the model. Defaults to None.
            num_of_layers (int): Number of layers in the autoencoder. Defaults to 3.
            layers_AE (list): List specifying the structure of layers in the autoencoder. Defaults to None.
            path_to_plot_model (str): Path where the plot model is saved. Defaults to an empty string.
            path_to_trajectories (str, optional): Path to the directory containing trajectory data. Defaults to an empty string.
            num_models (int, optional): Number of models to consider during training. Defaults to None.
            from_last (bool, optional): Whether to use models starting from the last in the directory. Defaults to False.
            prefix (str, optional): Prefix for identifying models in the directory. Defaults to 'model-'.
            every_nth (int, optional): Consider every nth trajectory model. Defaults to 1.
            grid_step (float, optional): Step size for grid generation in latent space. Defaults to 0.1.
            d_max_latent (float, optional): Maximum distance in latent space for grid scaling. Defaults to 2.0.
            anchor_mode (str, optional): Mode for anchor loss calculation. Defaults to 'circle'.
            rec_weight (float, optional): Weight for reconstruction loss. Defaults to 1.0.
            anchor_weight (float, optional): Weight for anchor loss. Defaults to 0.0.
            lastzero_weight (float, optional): Weight for last-zero loss. Defaults to 0.0.
            polars_weight (float, optional): Weight for polars loss. Defaults to 0.0.
            wellspacedtrajectory_weight (float, optional): Weight for well-spaced trajectory loss. Defaults to 0.0.
            gridscaling_weight (float, optional): Weight for grid-scaling loss. Defaults to 0.0.
        """
        
        self.mode = mode
        self.num_of_layers = num_of_layers
        self.layers_AE = layers_AE
        self.path_to_plot_model = path_to_plot_model
        self.latent_dim = 2

        # Data-related arguments
        self.path_to_trajectories = path_to_trajectories
        self.num_models = num_models
        self.from_last = from_last
        self.prefix = prefix
        self.every_nth = every_nth

        # Grid-related arguments
        self.grid_step = grid_step
        self.d_max_latent = d_max_latent
        self.anchor_mode = anchor_mode

        self.path_to_plot_model_directory = os.path.dirname(self.path_to_plot_model)
        if not os.path.exists(self.path_to_plot_model_directory):
            os.makedirs(self.path_to_plot_model_directory)

        # Convert args to JSON format
        args_dict = vars(self)  # Convert Namespace object to dictionary
        json_str = json.dumps(args_dict, indent=4)  # Convert dictionary to JSON string
        # Save JSON to file
        with open(os.path.join(self.path_to_plot_model_directory, 'args.json'), 'w') as f:
            f.write(json_str)

        # Weights
        self.loss_dict = {
                    'rec': {'official_name': "Reconstruction loss", 'weight': rec_weight},
                    'anchor': {'official_name': "Anchor loss", 'weight': anchor_weight},
                    'lastzero': {'official_name': "LastZero loss", 'weight': lastzero_weight},
                    'polars': {'official_name': "Polar loss", 'weight': polars_weight},
                    'gridscaling': {'official_name': "Grid-scaling loss", 'weight': gridscaling_weight},
                    'wellspacedtrajectory': {'official_name': "Well-spaced-trajectory loss", 'weight': wellspacedtrajectory_weight},
                }
        self.isEnabled = lambda loss: self.loss_dict[loss]['weight'] > 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_files_and_compile_train_mode(self, batch_size: int = 32):
        """Get models files anf complile for training process.
        
        Args: 
            batch_size (int, optional): Batch size for dataloader. Defaults to 32"""

        pt_files = get_files(self.path_to_trajectories, self.num_models, prefix=self.prefix, from_last=self.from_last, every_nth=self.every_nth)

        range_of_files_for_anchor = range(len(pt_files))


        rec_data_loader, transform = get_trajectory_dataloader(pt_files, batch_size, self.path_to_plot_model_directory)
        self.loss_dict['rec']['dataloader'] = rec_data_loader
        dataset = rec_data_loader.dataset
        input_dim = dataset[0].shape[0]
        print('number of models considered: ', len(dataset))


        if self.isEnabled('anchor'):
            anchor_dataloader = get_anchor_dataloader(dataset, range_of_files_for_anchor)
            self.loss_dict['anchor']['dataloader'] = anchor_dataloader
            predefined_values = get_predefined_values(anchor_dataloader.dataset, self.anchor_mode)
            self.predefined_values = predefined_values.to(self.device)

        if self.isEnabled('lastzero'):
            self.loss_dict['lastzero']['dataloader'] =  get_anchor_dataloader(dataset)
        if self.isEnabled('polars'):
            self.loss_dict['polars']['dataloader'] =  get_anchor_dataloader(dataset)
        l_max_inputspace=None
        if self.isEnabled('gridscaling'):
            self.loss_dict['gridscaling']['dataloader'] = get_gridpoint_and_trajectory_datasets(pt_files, self.path_to_plot_model_directory, self.grid_step, batch_size=batch_size)
            data_trajectory_dataset_temp = self.loss_dict['gridscaling']['dataloader'].dataset
            data_trajectory_dataset_temp_0 = data_trajectory_dataset_temp[0][1]
            data_trajectory_dataset_temp_last = data_trajectory_dataset_temp[-1][1]
            self.l_max_inputspace = torch.sqrt((data_trajectory_dataset_temp_0 - data_trajectory_dataset_temp_last).pow(2).sum(dim=-1)).to(self.device)
        if self.isEnabled('wellspacedtrajectory'):
            self.loss_dict['wellspacedtrajectory']['dataloader'], _ = get_trajectory_dataloader(pt_files, len(pt_files), self.path_to_plot_model_directory, shuffle=False)
        return input_dim
    

    def train(self, optimizer: Optimizer, epochs: int, every_epoch: int, batch_size: int, resume: bool, callbacks: Union[List, None] = None):

        """Train model.

        Args:
        optimizer (Optimizer): The optimizer object.
        epochs (int): The number of training epochs.
        every_epoch (int): The frequency (in epochs) at which callbacks are triggered and logs are saved.
        batch_size (int): The batch size for training.
        resume (bool): A flag indicating whether to resume training from an existing model. If a model exists but `resume=False`, an error is raised.
        callbacks (Union[List, None], optional): A list of callback objects used to manage the training process. Defaults to None.
        """

        input_dim = self.get_files_and_compile_train_mode(batch_size)
        self.AE_model = UniformAutoencoder(input_dim, self.num_of_layers, self.latent_dim, h=self.layers_AE).to(self.device)

        self.optimizer = optimizer.optimizer_choice(self.mode, self.AE_model)

        scheduler = optimizer.scheduler
        callbacks = CallbackList(callbacks=callbacks, model=self)

        def cycle_dataloader(dataloader):
            """Returns an infinite iterator for a dataloader."""
            return itertools.cycle(iter(dataloader))


        if (not os.path.exists(self.path_to_plot_model)) and (resume):
            raise "Can't resume without a model"
        
        best_AE_model = None
        if os.path.exists(self.path_to_plot_model):
            self.AE_model.load_state_dict(torch.load(self.path_to_plot_model, weights_only=True))
            best_AE_model = self.AE_model



        if (best_AE_model is not None) and (not resume):
            raise "There is a model already. Use --resume to update it."
        
        callbacks.on_train_begin()

        columns=['epoch']
        iterators = {}
        for i in self.loss_dict.keys():
            if self.isEnabled(i):
                iterators[i] = {
                    'iterator': iter(cycle_dataloader(self.loss_dict[i]['dataloader'])),
                    'maxbatch': len(self.loss_dict[i]['dataloader'])
                }
                columns.append(self.loss_dict[i]['official_name'])

        max_batches = max([iterators[d]['maxbatch'] for d in iterators.keys()])

        
        columns.append('Total loss')
        columns.append('Learning rate')
        df_losses = pd.DataFrame(columns=columns)
        for self.epoch in range(epochs):
            if callbacks.callbacks[0].stop_training == False:
                self.AE_model.train()
                total_losses = {}
                for i in self.loss_dict.keys():
                    total_losses[i] = 0
                total_loss = 0
            
                for batch_idx in range(max_batches):
                    self.optimizer.zero_grad()
                    losses = {}

                    data = {}
                    for i in self.loss_dict.keys():
                        if self.isEnabled(i):
                            data[i] = next(iterators[i]['iterator'])


                    if self.isEnabled('rec'):
                        data['rec'] = data['rec'].to(self.device).float()
                        x_recon, z = self.AE_model(data['rec'])
                        loss_t = 0
                        losses['rec'] = rec_loss_function(x_recon, data['rec'], z)

                    if self.isEnabled('anchor'):
                        data['anchor'] = data['anchor'].to(self.device).float()
                        x_recon, z = self.AE_model(data['anchor'])
                        losses['anchor'] = loss_anchor(z, self.predefined_values)
                    
                    if self.isEnabled('lastzero'):
                        data['lastzero'] = data['lastzero'].to(self.device).float()
                        x_recon, z = self.AE_model(data['lastzero'])
                        last_coordinate = z[-1, :]
                        loss_zero = torch.nn.functional.mse_loss(10*last_coordinate, torch.zeros_like(last_coordinate))
                        losses['lastzero'] = loss_zero

                    if self.isEnabled('polars'):
                        data['polars'] = data['polars'].to(self.device).float()
                        x_recon, z = self.AE_model(data['polars'])
                        last_coordinate = z[-1, :]
                        first_coordinate = z[0, :]
                        loss_zero = torch.nn.functional.mse_loss(10*last_coordinate, 10*0.8*torch.ones_like(last_coordinate))
                        loss_zero2 = torch.nn.functional.mse_loss(10*first_coordinate, 10*-0.8*torch.ones_like(first_coordinate))
                        losses['polars'] = loss_zero + loss_zero2

                    if self.isEnabled('wellspacedtrajectory'):
                        x_recon, z = self.AE_model(data['wellspacedtrajectory'].to(self.device))
                        losses['wellspacedtrajectory'] = loss_well_spaced_trajectory(z)


                    if self.isEnabled('gridscaling'):
                        data_grid_latent, data_trajectory = data['gridscaling']
                        data_grid_latent = data_grid_latent[0] # because of TesnorDataset
                        data_grid_latent = data_grid_latent.to(self.device)
                        data_trajectory = data_trajectory.to(self.device)
                        losses['gridscaling'] = loss_grid_to_trajectory(self.AE_model, data_grid_latent, data_trajectory, self.l_max_inputspace, epoch=self.epoch, d_max_latent=self.d_max_latent)

                    loss_total_batch = torch.tensor(0.0, dtype=torch.float32, device="cuda")
                    for i in losses:
                        weighted_loss = losses[i].float()*float(self.loss_dict[i]['weight'])
                        loss_total_batch += weighted_loss
                        total_losses[i] += weighted_loss.item()
                    total_loss += loss_total_batch.item()


                    loss_total_batch.backward()
                    self.optimizer.step()
                    scheduler.step(self.epoch + batch_idx/max_batches)
            else:
                break

            
            
            for i in losses:
                total_losses[i] = total_losses[i]/max_batches     
            self.total_loss = total_loss/max_batches


            row = {'epoch': self.epoch}
            for i in self.loss_dict.keys():
                row[self.loss_dict[i]['official_name']] = total_losses[i]
            row['Total loss'] = total_loss
            row['Learning rate'] = scheduler.get_last_lr()[0] 
            df_losses = pd.concat([df_losses, pd.DataFrame([row])], ignore_index=True)

            

            if self.epoch % every_epoch == 0:
                callbacks.on_epoch_end()

                printed_string = f"Epoch: {self.epoch}\t"
                for i in self.loss_dict:
                    if self.loss_dict[i]['weight']>0:
                        printed_string += f"{self.loss_dict[i]['official_name']}: {total_losses[i]:.4f}\t"
                printed_string += f"Total: {self.total_loss:.4f}"

                print(printed_string)

                df_losses.to_csv(os.path.join(self.path_to_plot_model_directory, 'losses.csv'), index=False)
                filtered_columns = ['epoch', 'Total loss']
                for i in losses.keys():
                    filtered_columns = filtered_columns + [self.loss_dict[i]['official_name']]
                
        plot_losses(df_losses[filtered_columns], every_epoch, self.path_to_plot_model_directory)
        best_AE_model = callbacks.on_train_end()