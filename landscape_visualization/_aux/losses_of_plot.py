# This code is partially based on the repository source: https://github.com/elhamod/NeuroVisualizer.git.
import torch
import torch.nn as nn
from landscape_visualization._aux.utils import get_closest_point_and_distance


####### LOSSES ###########


# reconstruction loss
def rec_loss_function(recon_x, x, z):
    """
    Calculates the reconstruction loss between the original input and its reconstruction using Mean Squared Error.
    This loss quantifies how well the neural network can reconstruct the input data after encoding and decoding,
    serving as a measure of the information preserved during the encoding process.
    
    Args:
        recon_x (torch.Tensor): The reconstructed input tensor.
        x (torch.Tensor): The original input tensor.
        z (torch.Tensor): The latent space representation (unused in this function).
    
    Returns:
        torch.Tensor: The reconstruction loss, a scalar tensor representing the sum of squared errors.
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    return recon_loss


# anchor loss
def loss_anchor(z, predefined_values):
    """
    Calculates the mean squared error loss between the neural network's output and the predefined values representing the differential equation's constraints. This loss quantifies how well the network's solution adheres to the known conditions.
    
        Args:
            z (torch.Tensor): The output tensor from the neural network, representing the approximated solution.
            predefined_values (torch.Tensor): The target tensor, containing the predefined values or constraints derived from the differential equation.
    
        Returns:
            torch.Tensor: The mean squared error loss, a scalar value indicating the discrepancy between the network's output and the expected values. This value is used to optimize the neural network during training, guiding it towards a solution that satisfies the differential equation.
    """
    l = nn.functional.mse_loss(z, predefined_values, reduction='sum')
    return l


# equi-distant trajectory loss
def loss_consecutive_coordinates(coordinates):
    """
    Calculates the loss based on the distances between consecutive coordinates.
    
        This method calculates the Mean Squared Error (MSE) loss between the squared distances of consecutive coordinate points and a dynamically adjusted maximum distance threshold.
        This loss encourages the model to generate coordinate sequences where the points are close to each other, contributing to a smooth and continuous solution trajectory of the differential equation.
    
        Args:
            coordinates (torch.Tensor): Tensor of coordinates representing the solution trajectory.
    
        Returns:
            torch.Tensor: The MSE loss between the distances and the maximum distance threshold.
    """
    distances = (10 * coordinates[1:] - 10 * coordinates[:-1]).pow(2).sum(-1)
    max_ = torch.tensor(2 * torch.pi * 10 * 0.8 / (coordinates.shape[0])).to(distances).pow(2)
    return nn.functional.mse_loss(distances, max_, reduction='sum')


# grid density loss
def loss_grid_to_trajectory(model, data_grid_latent, data_trajectory, l_max_inputspace, d_max_latent=2 ** 2, epoch=-1):
    """
    Calculates a loss that encourages the latent space to reflect the structure of the input space, penalizing deviations from a desired ratio of distances.
    
    This loss is computed by comparing distances between grid points and trajectory points in both the latent and input spaces. By minimizing this loss, the network learns a latent space where distances correspond to meaningful differences in the original data space, aiding in the neural approximation of differential equation solutions.
    
    Args:
        model: The neural network model, including encoder and decoder components.
        data_grid_latent: Latent representation of the grid data.
        data_trajectory: The original trajectory data.
        l_max_inputspace: The maximum distance in the input space, used for scaling.
        d_max_latent: The maximum distance in the latent space (default: 2**2).
        epoch: The current epoch number (default: -1). Used for printing information only on the first epoch.
    
    Returns:
        torch.Tensor: The calculated loss value.
    """
    _, data_trajectory_latent = model(data_trajectory)
    data_trajectory_latent = data_trajectory_latent.detach()  # NOTE: we only want grid points to affect, not trajectory points
    closest_trajectory_latent_points, closest_trajectory_latent_points_index, _ = get_closest_point_and_distance(
        data_grid_latent, data_trajectory_latent)
    data_grid_rec = model.decoder(data_grid_latent)

    l_inputspace = torch.sqrt(
        (data_grid_rec - data_trajectory[closest_trajectory_latent_points_index]).pow(2).sum(dim=-1))

    d_latentspace = torch.sqrt((data_grid_latent - closest_trajectory_latent_points).pow(2).sum(dim=-1))

    log_dist_ratio = torch.log(
        l_inputspace) - d_latentspace

    max_ = torch.log(l_max_inputspace) - d_max_latent
    if epoch == 0:
        print("loss_grid_to_trajectory: Automatic ratio calculated: " + str(max_.item()))

    loss = nn.functional.mse_loss(log_dist_ratio, max_, reduction='sum')
    return loss
