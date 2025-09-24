# This code is partially based on the repository source: https://github.com/elhamod/NeuroVisualizer.git.
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict

import torch
import builtins
import numpy as np

torch.serialization.add_safe_globals(
    [torch.nn.Sequential, torch.nn.modules.linear.Linear, torch.nn.modules.activation.Tanh, builtins.set])


def calculate_mean_std(saved_trajectories):
    """
    Calculates the mean and standard deviation of flattened vectors from a list of saved trajectories.
    
        This method processes a list of saved trajectories, which can either be a list of file paths
        to PyTorch state dictionaries or a list of OrderedDict objects representing state dictionaries.
        It computes the mean and standard deviation for each key in the state dictionaries and then
        concatenates these values into flattened vectors. This is done to aggregate information across multiple trained models
        or optimization states, providing a basis for analyzing the overall behavior and stability of the neural network-based
        differential equation solver.
    
        Args:
            saved_trajectories: A list of either file paths to PyTorch state dictionaries or OrderedDict objects
                representing the states of the neural network at different points during training or from different training runs.
    
        Returns:
            tuple: A tuple containing two torch.Tensor objects:
                - The first tensor is the mean of the flattened vectors, representing the average state of the network parameters.
                - The second tensor is the standard deviation of the flattened vectors, indicating the variability in the network parameters.
    """
    if isinstance(saved_trajectories[0], OrderedDict):
        state_dicts = saved_trajectories
        isSpecialCase = not isinstance(state_dicts[0], dict) if state_dicts else False
    else:
        state_dicts = [torch.load(file_path, map_location=torch.device('cpu'), weights_only=True) for file_path in
                       saved_trajectories]
        isSpecialCase = not isinstance(state_dicts[0], dict)

    keys = list(state_dicts[0].state_dict().keys() if isSpecialCase else state_dicts[0].keys())

    mean_values, std_values = [], []

    for key in keys:
        if isSpecialCase:
            values = [state_dict.state_dict()[key].float().view(1, -1) for state_dict in state_dicts]
        else:
            values = [state_dict[key].float().view(1, -1) for state_dict in state_dicts]

        values_st = torch.stack(values)

        mean = torch.mean(values_st, dim=0)
        std = torch.std(values_st, dim=0)

        mean_values.append(mean)
        std_values.append(std)

    mean_flattened_vector = torch.cat(mean_values, dim=1).view(-1)
    std_flattened_vector = torch.cat(std_values, dim=1).view(-1)

    return mean_flattened_vector, std_flattened_vector


class NormalizeModelParameters:
    """
    Normalizes model parameters using pre-computed mean and standard deviation.
    
        This class applies a normalization transformation to model parameters,
        subtracting a mean and dividing by a standard deviation. It's typically
        used to standardize parameters before or during optimization.
    
        Attributes:
            mean: The mean value used for normalization.
            std: The standard deviation used for normalization.
            eps: A small constant added to the standard deviation for numerical stability.
    """

    def __init__(self, mean, std):
        """
        Initializes a Gaussian distribution object for parameter normalization.
        
                This method initializes a Gaussian distribution with a specified
                mean and standard deviation. It's used to normalize the parameters
                of the neural network model, which can improve training stability
                and convergence when solving differential equations.
        
                Args:
                    mean (float): The mean of the Gaussian distribution.
                    std (float): The standard deviation of the Gaussian distribution.
        
                Returns:
                    None
        """
        self.mean = mean
        self.std = std

    def __call__(self, parameters):
        """
        Applies standardization to the input parameters, ensuring consistent scaling for neural network training.
        
                It subtracts the precomputed mean and divides by the precomputed standard deviation, adding a small epsilon value for numerical stability. This normalization step is crucial for optimizing the training process of neural networks used to solve differential equations.
        
                Args:
                    parameters (torch.Tensor): The input tensor to be standardized.
        
                Returns:
                    torch.Tensor: The standardized tensor.
        """
        return (parameters - self.mean) / (self.std + torch.finfo(torch.float32).eps)


class ModelParamsDataset(Dataset):
    """
    Dataset that loads model parameters from trajectories.
    
        This dataset is designed to load model parameters from a collection of trajectories,
        potentially applying a transformation to the parameters. It supports loading from
        both in-memory trajectories (lists of OrderedDicts) and trajectories stored as
        separate files.
    
        Class Methods:
        - __init__:
    """

    def __init__(self, saved_trajectories, transform=None):
        """
        Initializes the ModelParamsDataset object.
        
        This class stores the trajectories and applies a transform if provided,
        preparing the data for training a neural network to solve differential equations.
        The transform can be used to normalize or augment the trajectories.
        
        Args:
            saved_trajectories: The trajectories to be saved. These trajectories represent the data
                that the neural network will learn from to approximate the solution of a
                differential equation.
            transform: An optional transform to apply to the trajectories. This allows for
                preprocessing steps such as normalization or data augmentation to be applied
                before training the neural network.
        
        Returns:
            None.
        
        Why:
            This initialization is crucial for setting up the dataset that will be used to train
            the neural network. By storing the trajectories and applying a transform, the data
            is prepared in a format suitable for learning the underlying dynamics of the
            differential equation.
        """
        self.saved_trajectories = saved_trajectories
        self.transform = transform
        self.check = None

    def __len__(self):
        """
        Returns the number of trajectories stored in the dataset. This reflects the amount of data available for training or evaluation of the neural network model approximating the differential equation solution.
        
                Args:
                    self: The ModelParamsDataset instance.
        
                Returns:
                    int: The number of saved trajectories.
        """
        return len(self.saved_trajectories)

    def __getitem__(self, idx):
        """
        Retrieves a trajectory (parameter vector) from the dataset.
        
                This method is crucial for accessing individual trajectories, which represent
                the model's parameters at different points in its training history.
                It supports loading trajectories stored as either lists of OrderedDicts
                (representing model states) or as individual files. The parameters
                are extracted and concatenated into a single tensor, which can then be
                transformed if a transformation function is provided. This allows to analyze model's evolution during training process.
        
                Args:
                    idx (int): The index of the trajectory to retrieve.
        
                Returns:
                    torch.Tensor: A tensor containing the concatenated parameters of the trajectory,
                        potentially transformed.
        """
        if isinstance(self.saved_trajectories[0], OrderedDict):
            model_dict = self.saved_trajectories[idx]
            params = [value.float().view(-1) for value in model_dict.values()]
        else:
            file_path = self.saved_trajectories[idx]
            model_dict = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)
            isSpecialCase = not hasattr(model_dict, 'keys')
            if isSpecialCase:
                model_dict = model_dict.state_dict()
            params = []
            for param_tensor in model_dict:
                params.append(model_dict[param_tensor].flatten())

        data = torch.cat(params)

        if self.transform:
            data = self.transform(data)

        return data


def get_trajectory_dataset(saved_trajectories, normalize=True):
    """
    Creates a dataset of trajectories and a normalizer for training neural differential equation solvers.
    
        This method processes a list of trajectory data by computing the mean and
        standard deviation of trajectory parameters. It then constructs a dataset
        suitable for training neural networks to approximate solutions of differential
        equations. The normalizer is used to scale the input data, which can improve
        the training stability and convergence of the neural network.
    
        Args:
            saved_trajectories: A list of saved trajectories, where each trajectory
                represents a sequence of states or parameter values over time.
            normalize: A boolean indicating whether to normalize the data using the
                calculated mean and standard deviation. Defaults to True. Normalization
                is crucial for ensuring stable and efficient training of neural networks
                used to solve differential equations.
    
        Returns:
            tuple: A tuple containing the trajectory dataset and the normalizer.
                The first element is a ModelParamsDataset object, which is a PyTorch
                dataset that can be used for training or evaluation.
                The second element is a NormalizeModelParameters object, which is used
                to normalize the data. The normalizer is returned so that it can be
                used to normalize new data at inference time.
    """
    mean, std = calculate_mean_std(saved_trajectories)
    normalizer = NormalizeModelParameters(mean, std)
    return ModelParamsDataset(saved_trajectories, transform=normalizer if normalize else None), normalizer


def get_trajectory_dataloader(batch_size, models=None, pt_files=None, normalize=True, shuffle=True, device=None):
    """
    Creates a trajectory dataloader and normalizer for training neural network-based differential equation solvers.
    
        This function prepares trajectory data for efficient training by creating a dataloader and applying normalization.
        The dataloader streamlines the process of feeding data to the neural network, while normalization enhances training stability and convergence.
    
        Args:
            batch_size: The batch size for the dataloader.
            models: A list of models to use for creating the dataset. If None, pt_files must be provided.
            pt_files: A list of .pt files to use for creating the dataset. If None, models must be provided.
            normalize: Whether to normalize the trajectory data.
            shuffle: Whether to shuffle the data in the dataloader.
            device: The device to use for the data generator (e.g., 'cuda', 'mps').
    
        Returns:
            A tuple containing the dataloader and the normalizer. The dataloader is used to iterate over the trajectory data in batches during training,
            and the normalizer is used to scale the data to a suitable range for the neural network.
    """
    dataset, normalizer = get_trajectory_dataset(models if models else pt_files, normalize=normalize)

    if models:
        generator = torch.Generator(device=device)  # CUDA mps
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader, normalizer


def get_anchor_dataloader(dataset, subset=None):
    """
    Creates a DataLoader for a subset of the input dataset, used to prepare data for training neural network-based differential equation solvers.
    
        This method generates a DataLoader containing a specified subset of the
        provided dataset. If no subset is specified, the entire dataset is used.
        The DataLoader is configured to load the entire subset in a single batch
        without shuffling. This is done to ensure that the entire subset is available
        for calculating the loss and updating the network's weights in each training
        iteration.
    
        Args:
            dataset: The dataset to create a DataLoader from.
            subset: An optional list or range of indices specifying the subset of the
                dataset to use. If None, the entire dataset is used.
    
        Returns:
            DataLoader: A DataLoader containing the specified subset of the dataset.
                The batch size is equal to the length of the subset, and shuffling is
                disabled.
    """
    if subset is None:
        subset = range(len(dataset))
    dataset2 = torch.utils.data.Subset(dataset, subset)
    data_loader2 = DataLoader(dataset2, batch_size=len(dataset2), shuffle=False)
    print('number of constrained models considered: ', len(dataset2))

    return data_loader2


def get_predefined_values(dataset, anchor_mode="diagonal"):
    """
    Generates predefined anchor values based on the specified anchor mode.
    
        This method calculates and returns a set of predefined anchor values,
        which serve as initial guesses or constraints for the neural network
        during the differential equation solving process. The `anchor_mode`
        determines the distribution of these values, influencing the network's
        exploration of the solution space. By strategically initializing the
        solution space, we guide the neural network towards more accurate and
        stable solutions of the differential equation.
    
        Args:
          dataset: The input dataset used to determine the size or number of
            predefined values.
          anchor_mode: The mode for generating anchor values. Can be "diagonal"
            or "circle". Defaults to "diagonal".
    
        Returns:
          torch.Tensor: A tensor containing the predefined anchor values. The
            shape and content of the tensor depend on the selected anchor mode.
            For "diagonal" mode, it returns a tensor of tuples where x == y.
            For "circle" mode, it returns a tensor of (x, y) coordinates
            representing points on a circle.
    """
    if anchor_mode == "diagonal":
        # Predefined set of values
        begin = -1.0  # endpoint of the array
        end = 1.0
        s = len(dataset)  # number of steps between start and endpoint

        # Create an array of s+2 points with [0.0, 0.0] as the first point and [n, n] as the last point
        points = np.linspace(begin, end, s)
        # Create an array of tuples from the points array
        tuples_array = np.array([(x, y) for x in points for y in points])
        # Filter the tuples array to only include tuples within the range of [0.0, 0.0] and [n, n]
        tuples_array = tuples_array[
            (tuples_array[:, 0] <= end) & (tuples_array[:, 1] <= end) & (tuples_array[:, 0] == tuples_array[:, 1])]
        predefined_values = torch.tensor(tuples_array, dtype=torch.float32)
    elif anchor_mode == "circle":
        # Example data
        n = len(dataset)  # Number of points on the circle
        r = 0.8  # Radius of the circle

        # Generate coordinates
        theta = torch.linspace(0, 2 * torch.pi, n + 1)[:-1]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        predefined_values = torch.stack([x, y], dim=1)

    else:
        raise "anchor_mode not implemented"
    return predefined_values
