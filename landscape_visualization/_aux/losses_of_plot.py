# This code is partially based on the repository source: https://github.com/elhamod/NeuroVisualizer.git.
import torch
import torch.nn as nn

from landscape_visualization._aux.utils import get_closest_point_and_distance

####### LOSSES ###########


# reconstruction loss
def rec_loss_function(recon_x, x, z):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') 
    return recon_loss 

# anchor loss
def loss_anchor(z, predefined_values):
    l = nn.functional.mse_loss(z, predefined_values, reduction='sum')
    return l

# equi-distant trajectory loss
def loss_consecutive_coordinates(coordinates):
    distances = (10*coordinates[1:] - 10*coordinates[:-1]).pow(2).sum(-1)
    max_ = torch.tensor(2*torch.pi*10*0.8/(coordinates.shape[0])).to(distances).pow(2)
    return nn.functional.mse_loss(distances, max_, reduction='sum')

    

# grid density loss
def loss_grid_to_trajectory(model, data_grid_latent, data_trajectory, l_max_inputspace, d_max_latent=2**2, epoch=-1):    
    _, data_trajectory_latent = model(data_trajectory)
    data_trajectory_latent = data_trajectory_latent.detach() # NOTE: we only want grid points to affect, not trajectory points
    closest_trajectory_latent_points, closest_trajectory_latent_points_index, _ = get_closest_point_and_distance(data_grid_latent, data_trajectory_latent)
    data_grid_rec = model.decoder(data_grid_latent)

    l_inputspace = torch.sqrt((data_grid_rec - data_trajectory[closest_trajectory_latent_points_index]).pow(2).sum(dim=-1))

    d_latentspace =  torch.sqrt((data_grid_latent- closest_trajectory_latent_points).pow(2).sum(dim=-1))

    log_dist_ratio = torch.log(l_inputspace) - d_latentspace

    max_ = torch.log(l_max_inputspace) - d_max_latent
    if epoch == 0:
        print("loss_grid_to_trajectory: Automatic ratio calculated: " +str(max_.item()))

    loss = nn.functional.mse_loss(log_dist_ratio, max_, reduction='sum') 

    return loss





