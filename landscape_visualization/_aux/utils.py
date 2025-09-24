# This code is partially based on the repository source: https://github.com/elhamod/NeuroVisualizer.git.

import re
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from landscape_visualization._aux.trajectories_data import get_trajectory_dataset


#### plotting
def plot_losses(df, every_epoch, file_path):
    """
    Plots the loss values during the training of a neural network-based differential equation solver and saves the plot to a file.
    
        This method visualizes the training progress by generating a line plot of the loss values for each loss component in the input DataFrame against the epoch number. The plot is saved as a PDF file, providing a visual aid for assessing the convergence and stability of the training process. This is crucial for understanding how well the neural network is learning to approximate the solution to the differential equation.
    
        Args:
            df: DataFrame containing the loss values, with an 'epoch' column
                and columns for each loss to be plotted. Each column represents a different loss component during training.
            every_epoch: Integer representing the interval at which to display
                epoch ticks on the x-axis. This helps in decluttering the plot and improving readability, especially for long training runs.
            file_path: String representing the directory where the plot should be saved.
                Specifies the location where the generated plot will be stored.
    
        Returns:
            None. The function saves the plot to a file, allowing for later analysis and comparison of different training runs.
    """
    plt.figure()

    # Set x and y axis labels
    x_label = 'Epoch'
    y_label = 'Loss'

    # Set plot style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create plot
    fig, ax = plt.subplots()

    # Plot each column as a line plot
    for column in df.columns:
        if column != 'epoch':
            ax.plot(df['epoch'], df[column], label=column)

    # Set x and y axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Set x-axis tick intervals
    ax.set_xticks(df['epoch'][::every_epoch])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

    # Set log scale on y-axis
    plt.yscale('log')

    # Set legend
    ax.legend()

    # Save plot to PNG file with 300 dpi
    plt.savefig(os.path.join(file_path, 'losses.pdf'), dpi=300)



####### Model stuff ####
def repopulate_model(flattened_params, new_model):
    """
    Populates a model's parameters from a flattened parameter vector, enabling the exploration of the model's parameter space.
    
        This method takes a flattened parameter vector and a model, and updates the
        model's parameters with the values from the flattened vector. It iterates
        through the model's state dictionary, extracting the size of each parameter
        tensor and copying the corresponding slice from the flattened vector into
        the parameter tensor. This is crucial for operations like gradient-free optimization or ensemble methods, where different parameter sets need to be rapidly evaluated.
    
        Args:
            flattened_params (torch.Tensor): A flattened vector containing the model's parameters.
            new_model (torch.nn.Module): The model to be populated with the flattened parameters.
    
        Returns:
            torch.nn.Module: The model with updated parameters.
    """
    start_idx = 0
    state_dict = new_model.state_dict()
    for name, param in enumerate(state_dict):
        param = state_dict[param]
        size = param.numel()
        sub_flattened = flattened_params[start_idx: start_idx + size].view(param.size())
        param.data.copy_(sub_flattened)
        start_idx += size
    return new_model


def get_closest_point_and_distance(grid_points, trajectory_points):
    """
    Finds the closest point on the trajectory to each point on the spatial grid.
    
        This function is crucial for evaluating the neural network's approximation
        of the differential equation's solution. By finding the closest point on
        the trajectory to each grid point, we can assess how well the neural network
        solution aligns with the expected behavior in the spatial domain. It computes
        the distances between grid points and trajectory points, identifies the closest
        trajectory point for each grid point, and returns the closest points, their
        indices, and the distances to them.
    
        Args:
            grid_points (torch.Tensor): A tensor of grid points representing the spatial domain.
            trajectory_points (torch.Tensor): A tensor of trajectory points representing the solution trajectory.
    
        Returns:
            tuple: A tuple containing:
                - closest_trajectory_points (torch.Tensor): A tensor of the closest trajectory points to each grid point.
                - closest_trajectory_points_index (torch.Tensor): A tensor of the indices of the closest trajectory points.
                - distance_from_closest_trajectory (torch.Tensor): A tensor of the distances from each grid point to its closest trajectory point.
    """
    # Compute the distances between each point in tensor1 and tensor2
    distances = torch.cdist(grid_points, trajectory_points)

    # Find the index of the closest point in tensor2 for each point in tensor1
    closest_trajectory_points_index = torch.argmin(distances, dim=1)

    # Extract the closest points from tensor2 for each point in tensor1
    closest_trajectory_points = trajectory_points[closest_trajectory_points_index]

    # Compute the distances between each point in tensor1 and its closest point in tensor2
    distance_from_closest_trajectory = torch.gather(distances, 1, closest_trajectory_points_index.unsqueeze(1))

    return closest_trajectory_points, closest_trajectory_points_index, distance_from_closest_trajectory


##### Misc

def get_files(file_path, num_models=None, prefix="", from_last=False, every_nth=1):
    """
    Retrieves a list of '.pt' files, representing trained neural network models, from a specified directory. These models are used to approximate solutions to differential equations. The function allows for filtering and ordering these files to select specific models for evaluation or further training.
    
        Args:
            file_path (str): The path to the directory containing the '.pt' files.
            num_models (int, optional): The maximum number of model files to return. If None, all files are returned. Defaults to None.
            prefix (str, optional): A prefix used to extract the numerical part of the filename for sorting. This is useful for ensuring models are loaded in the correct sequence based on training iteration or epoch. Defaults to "".
            from_last (bool, optional): If True, returns the last 'num_models' files; otherwise, returns the first 'num_models' files. This allows for selecting the most recently trained models. Defaults to False.
            every_nth (int, optional): Returns every nth file. If greater than 1, it also ensures the last file is included. This can be used to sample models from different stages of training. Defaults to 1.
    
        Returns:
            list: A list of file paths ending with '.pt', sorted numerically based on the filename (if possible), and potentially truncated to 'num_models'. The order and selection of these files are crucial for analyzing the training progress and performance of the neural network solver.
    """
    def extract_number(s, prefix=prefix):
        pattern = re.compile(r'{}(\d+).pt'.format(prefix))
        match = pattern.search(s)
        if match:
            return int(match.group(1))
        else:
            return float('inf')

    def get_all_files(d):
        f_ = []
        for dirpath, dirnames, filenames in os.walk(d):
            f_temp = []
            for filename in filenames:
                f_temp.append(os.path.join(dirpath, filename))

            f_temp = [file for file in f_temp if os.path.splitext(file)[-1] == ".pt"]
            f_temp = sorted(f_temp, key=extract_number)

            len_f_temp_original = len(f_temp)

            if every_nth > 1 and len_f_temp_original > 0:
                f_temp_last = f_temp[-1]
                f_temp = f_temp[::every_nth]

                if len_f_temp_original % every_nth != 1:
                    f_temp = f_temp + [f_temp_last]

            f_ = f_ + f_temp
        return f_

    directory = os.path.join(file_path)
    files = get_all_files(directory)
    pt_files = [file for file in files if file.endswith(".pt")]
    print(len(pt_files), 'files included.')
    if num_models is not None:
        pt_files = pt_files[:num_models] if not from_last else pt_files[-num_models:]

    return pt_files


def get_gridpoint_dataset(grid_step=0.1):
    """
    Generates a gridpoint dataset for evaluating the neural network solution.
    
    Generates a dataset of grid points within a specified range to evaluate the trained neural network's approximation of the differential equation's solution across the domain. This is done by creating a uniform grid of points that span the domain of interest.
    
    Args:
        grid_step (float, optional): The step size for generating the grid. Smaller values result in a finer grid and more evaluation points. Defaults to 0.1.
    
    Returns:
        TensorDataset: A TensorDataset containing the grid points, where each point represents a location at which the neural network's solution will be evaluated.
    """
    min_grid = -1.
    max_grid = 1.
    # Generate a 1D tensor of x values
    x_values = torch.arange(min_grid, max_grid, grid_step)
    # Generate a 1D tensor of y values
    y_values = torch.arange(min_grid, max_grid, grid_step)
    # Use meshgrid and stack to generate a 2D tensor of all coordinates on the grid
    X, Y = torch.meshgrid(x_values, y_values)
    grid_tensor = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    # Create a TensorDataset from the tensor
    dataset = TensorDataset(grid_tensor)
    # Create a DataLoader from the dataset
    return dataset


class PaddedConcatDataset(Dataset):
    """
    A dataset that concatenates two datasets, padding the shorter one by cycling through it.
    
        Class Methods:
        - __init__
        - __getitem__
        - __len__
    """

    def __init__(self, dataset1, dataset2):
        """
        Initializes a new instance of the PaddedConcatDataset class.
        
                This method initializes the object by associating it with two datasets.
                This setup is crucial for managing and processing multiple datasets in a unified manner,
                allowing the neural network to learn from a more comprehensive set of examples when solving differential equations.
        
                Args:
                    dataset1: The first dataset to be associated with the object.
                    dataset2: The second dataset to be associated with the object.
        
                Returns:
                    None.
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, idx):
        """
        Retrieves a pair of data samples, ensuring both datasets contribute to training the neural network model for solving differential equations.
        
                This method retrieves corresponding data points from two datasets. When the datasets have unequal sizes, it addresses the discrepancy by cycling through the shorter dataset. This ensures that each dataset contributes equally during the training process of the neural network, preventing bias towards the larger dataset and optimizing the learning of the differential equation's solution.
        
                Args:
                    idx (int): The index of the desired data pair.
        
                Returns:
                    tuple: A tuple containing a data sample from `dataset1` and a data sample from `dataset2`.
                           These samples are used as inputs for training the neural network to approximate the solution
                           of a differential equation.
        """
        if idx < len(self.dataset1):
            data1 = self.dataset1[idx]
            if idx < len(self.dataset2):
                data2 = self.dataset2[idx]
            else:
                # Pad the smaller dataset with its own samples
                data2 = self.dataset2[idx % len(self.dataset2)]
        else:
            data2 = self.dataset2[idx]
            if idx < len(self.dataset1):
                data1 = self.dataset1[idx]
            else:
                # Pad the smaller dataset with its own samples
                data1 = self.dataset1[idx % len(self.dataset1)]

        return (data1, data2)

    def __len__(self):
        """
        Returns the length of the longest dataset.
        
                To ensure consistent batching and training, the length of the
                `PaddedConcatDataset` is defined by its longest component dataset.
                This method returns the length of the longer dataset between
                `dataset1` and `dataset2`, which is crucial for proper data handling
                during the training of neural network models for solving
                differential equations.
        
                Args:
                    self: The object instance.
        
                Returns:
                    int: The length of the longest dataset.
        """
        return max(len(self.dataset1), len(self.dataset2))


def get_gridpoint_and_trajectory_datasets(pt_files, path, grid_step=0.1, batch_size=32):
    """
    Combines preprocessed data representing both grid points and trajectory data into a unified DataLoader for efficient training.
    
        This function prepares data for training a neural network to solve differential equations.
        It merges the data obtained from discretizing the domain (gridpoint dataset) with trajectory data,
        creating a comprehensive dataset suitable for learning the solution behavior. The combined
        dataset is then loaded into a DataLoader for batched training.
    
        Args:
            pt_files (list): List of .pt files containing trajectory data.
            path (str): Path to the directory containing the trajectory .pt files.
            grid_step (float, optional): Discretization step size for generating the gridpoint dataset. Defaults to 0.1.
            batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
    
        Returns:
            DataLoader: A DataLoader containing the concatenated and batched gridpoint and trajectory datasets,
                ready for training a neural network-based differential equation solver.
    """
    dataset_gridpoint = get_gridpoint_dataset(grid_step=grid_step)
    dataset_trajectory, _ = get_trajectory_dataset(pt_files, path)

    concat_dataset = PaddedConcatDataset(dataset_gridpoint, dataset_trajectory)

    dataloader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=False)

    return dataloader


def get_diagonal_mask(n, offset=1):
    """
    Returns a Boolean tensor of shape (n, n) highlighting regions adjacent to the diagonal.
        This mask is useful for defining specific interaction neighborhoods within the differential equation solver,
        allowing targeted computation and analysis around the diagonal elements.
    
        Args:
            n (int): The size of the tensor (n x n).
            offset (int, optional): The offset from the main diagonal. Positive values shift the diagonal upwards,
                negative values shift it downwards. Defaults to 1.
    
        Returns:
            torch.Tensor: A Boolean tensor of shape (n, n) with True values at the specified diagonal offset and False elsewhere.
    """
    diag = torch.ones(n - abs(offset), dtype=torch.bool)
    eye = torch.diag(diag, diagonal=offset)
    return eye


def loss_well_spaced_trajectory(coords):
    """
    Computes a loss that penalizes trajectories that are not well-spaced, ensuring the neural network learns smooth and physically plausible solutions to differential equations.
    
        This method calculates a loss based on the distances between coordinates in a trajectory.
        It encourages the trajectory to be well-spaced by penalizing points that are closer to
        other points than their immediate neighbors. This helps the neural network avoid generating erratic or discontinuous solutions when approximating the solution of a differential equation.
    
        Args:
            coords: A tensor of coordinates representing the trajectory. These coordinates represent the predicted solution path of the differential equation.
    
        Returns:
            torch.Tensor: A scalar tensor representing the loss value. This loss is minimized during training to encourage well-spaced and smooth trajectory predictions.
    """
    # Compute distances between adjacent coordinates
    prev_dists = torch.norm(coords[1:] - coords[:-1], dim=1)
    prev_dists = torch.cat([torch.tensor([0]).to(prev_dists), prev_dists], dim=0).unsqueeze(1)
    next_dists = torch.norm(coords[:-1] - coords[1:], dim=1)
    next_dists = torch.cat([next_dists, torch.tensor([0]).to(next_dists)], dim=0).unsqueeze(1)

    # Compute pair-wise distances
    pairwise_dists = torch.cdist(coords, coords)

    # Compute condition tensor
    condition_next = (pairwise_dists < next_dists)
    condition_prev = (pairwise_dists < prev_dists)

    where_closer_next = torch.where(condition_next, next_dists - pairwise_dists, torch.tensor([0.0]).to(next_dists))
    where_closer_previous = torch.where(condition_prev, prev_dists - pairwise_dists, torch.tensor([0.0]).to(next_dists))

    # Create mask tensor
    mask_previous = get_diagonal_mask(where_closer_previous.shape[0], offset=1).to(condition_next)
    mask_next = get_diagonal_mask(where_closer_next.shape[0], offset=-1).to(condition_next)
    mask_diag = get_diagonal_mask(where_closer_next.shape[0], offset=0).to(condition_next)
    mask = ~(mask_previous | mask_next | mask_diag)

    where_closer = torch.where(mask, where_closer_previous + where_closer_next, torch.tensor([0.0]).to(next_dists))

    loss = torch.sum(where_closer)

    return loss


##################################
# Density measures
##################################


def get_density(grid, type="inverse", p=1):
    """
    Calculates a measure of how well the neural network solution satisfies the differential equation within the given grid. Different density types offer various perspectives on solution accuracy.
    
        Args:
            grid: The spatial or temporal grid over which the differential equation is being solved.
            type: Specifies the method for calculating density. Options include "inverse" (default), "cos", and "CKA". Each type offers a different sensitivity to errors in the solution.
            p: A parameter used for the "inverse" density calculation, influencing the weighting of errors.
    
        Returns:
            The calculated density value, representing the overall accuracy of the solution on the grid. Higher density typically indicates a more accurate solution.
    
        Raises:
            Exception: If the specified density measure is not implemented, indicating an unsupported method for evaluating the solution.
    """
    if type == "inverse":
        return get_density_inverse(grid, p)
    elif type == "cos":
        return get_density_COS(grid)
    elif type == "CKA":
        return get_density_CKA(grid)
    else:
        raise "density measure not implemented"


# CKA similiarity
def linear_kernel(X, Y):
    """
    Computes the linear kernel between two matrices.
    
    This function calculates the linear kernel, which represents the dot product
    between the input matrices. This is a fundamental operation for
    assessing the similarity between data points in a high-dimensional space,
    which is useful when transforming the data into a feature space where
    linear relationships can be exploited to approximate the solution of
    differential equations.
    
    Args:
        X (np.ndarray): The first matrix of shape (n x m).
        Y (np.ndarray): The second matrix of shape (n x m).
    
    Returns:
        np.ndarray: The linear kernel matrix of shape (n x n).
    """
    # X: n x m
    # Y: n x m
    return np.dot(X, Y.T)
    # Returns: n x n


def centered_kernel_matrix(matrix):
    """
    Computes the centered kernel matrix for use in neural differential equation solvers.
    
    This method takes a data matrix as input, computes the linear kernel,
    and then centers the kernel matrix using the centering matrix H. Centering the kernel
    is a crucial step in some kernel methods used to solve differential equations with neural networks,
    as it can improve the conditioning and stability of the solution.
    
    Args:
        matrix (np.ndarray): The input data matrix (n x m), where n is the number of samples and m is the number of features.
    
    Returns:
        np.ndarray: The centered kernel matrix (n x n).
    """
    # matrix: n x m
    K = linear_kernel(matrix, matrix)
    # K: n x n

    n = K.shape[0]
    ones = np.ones((n, n)) / n
    H = np.eye(n) - ones
    HKH = H @ K @ H
    # HKH: n x n

    return HKH


def get_density_CKA(matrix):
    """
    Computes the density CKA (Centered Kernel Alignment) for a given matrix.
    
        This method calculates the density CKA based on the centered kernel matrix of the input.
        It normalizes the centered kernel matrix and then sums the resulting CKA matrix along one axis
        to obtain a density measure, reflecting the similarity between different states encountered
        during the differential equation solving process. This density can be used to analyze the
        exploration of the state space by the neural network solver.
    
        Args:
            matrix (np.ndarray): The input matrix (n x m) representing the states for which to compute the density CKA.
    
        Returns:
            np.ndarray: The density CKA for each row of the input matrix (n x 1), representing the similarity density.
    """
    # matrix: n x m
    HKH = centered_kernel_matrix(matrix)
    # HKH: n x n

    norms = np.sqrt(np.abs(np.diag(HKH)))
    # norms: n

    outer_norms = np.outer(norms, norms)
    # outer_norms: n x n

    cka_matrix = HKH / outer_norms
    cka_matrix = (cka_matrix + 1) / 2
    cka_matrix = np.clip(cka_matrix, 0, 1)
    # cka_matrix: n x n

    density = np.sum(cka_matrix, axis=1, keepdims=True)

    return density


def get_density_COS(grid):
    """
    Calculates the density of each item based on cosine similarity.
    
        This method quantifies the relationships between data points in the grid by
        assessing their similarity. By calculating the cosine similarity between
        normalized data points, the method determines how closely each point relates
        to others in the dataset. The density is then derived from the sum of these
        similarities, providing a measure of how interconnected each point is within
        the grid. This approach is useful for understanding the structure and
        relationships within the data, which is crucial for training neural networks
        to solve differential equations.
    
        Args:
            grid (np.ndarray): The input grid of data, where each row represents an
                item and each column represents a feature.
    
        Returns:
            np.ndarray: The density of each item in the grid, representing the sum
                of cosine similarities between that item and all other items.
    """
    normalized_grid = normalize(grid, axis=1)
    cosine_similarities = cosine_similarity(normalized_grid)
    cosine_similarities = np.abs(np.clip(cosine_similarities, -1, 1))
    weights = cosine_similarities

    density = np.sum(weights, axis=1, keepdims=True)

    return density


def get_density_inverse(grid, power=1):
    """
    Calculates a density measure for each point in the grid, based on inverse distances.
    
        This method approximates a solution by considering the relationships between points in a spatial grid.
        It computes a density value for each point, where the density is influenced by the inverse distance to all other points.
        This approach leverages the spatial arrangement of points to inform the approximation,
        akin to how neural networks in this project learn relationships to solve differential equations.
        The density is calculated using the inverse of the Euclidean distance raised to a specified power.
    
        Args:
            grid (numpy.ndarray): A numpy array representing the grid of points.
    
        Returns:
            numpy.ndarray: A numpy array containing the density for each point in the grid.
    """
    radius = None

    # calculate distances between each data point and all other data points
    distances = (grid[:, np.newaxis, :] - grid).astype(np.float16) ** 2
    distances = np.sum(distances, axis=2)
    distances = np.sqrt(distances)
    distances = np.where(distances == 0, np.inf, distances)

    # apply search radius if specified
    if radius is not None:
        distances[distances > radius] = np.inf

    # calculate weights based on the inverse distance
    weights = 1.0 / distances ** power

    density = np.sum(weights, axis=1, keepdims=True)

    return density
