import torch.cuda

from tedeous.optimizers.optimizer import Optimizer
from landscape_visualization._aux.visualization_model import VisualizationModel
from landscape_visualization._aux.early_stopping_plot import EarlyStopping

# print(torch.cuda.is_available())

if __name__ == '__main__':

    model_args = {
    "mode": "NN",
    "num_of_layers": 3,
    "layers_AE": [
        991,
        125,
        15
    ],
    "path_to_plot_model": "landscape_visualization_origin/saved_models/PINN_burgers_adam_5_starts/model.pt",
    "num_models": None,
    "from_last": False,
    "prefix": "model-",
    "path_to_trajectories": "landscape_visualization_origin/trajectories/burgers/adam_5_starts",
    "every_nth": 1,
    "grid_step": 0.1,
    "d_max_latent": 2,
    "anchor_mode": "circle",
    "rec_weight": 10000.0,
    "anchor_weight": 0.0,
    "lastzero_weight": 0.0,
    "polars_weight": 0.0,
    "wellspacedtrajectory_weight": 0.0,
    "gridscaling_weight": 0.0,}

    batch_size = 32
    epochs = 600000
    patience_scheduler = 400000
    every_epoch = 100
    cosine_scheduler_patience = 2000
    learning_rate = 0.0005
    resume = True

    model = VisualizationModel(**model_args)
    optim = Optimizer('RMSprop', {'lr': learning_rate}, cosine_scheduler_patience=cosine_scheduler_patience)
    cb_es = EarlyStopping(patience=patience_scheduler)

    model.train(optim, epochs, every_epoch, batch_size, resume, callbacks=[cb_es])
