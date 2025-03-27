# This code is partially based on the repository source: https://github.com/elhamod/NeuroVisualizer.git.
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict

import torch
import builtins
import numpy as np

torch.serialization.add_safe_globals(
    [torch.nn.Sequential, torch.nn.modules.linear.Linear, torch.nn.modules.activation.Tanh, builtins.set])


def calculate_mean_std(saved_trajectories):
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
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, parameters):
        return (parameters - self.mean) / (self.std + torch.finfo(torch.float32).eps)


class ModelParamsDataset(Dataset):
    def __init__(self, saved_trajectories, transform=None):
        self.saved_trajectories = saved_trajectories
        self.transform = transform
        self.check = None

    def __len__(self):
        return len(self.saved_trajectories)

    def __getitem__(self, idx):
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
    mean, std = calculate_mean_std(saved_trajectories)
    normalizer = NormalizeModelParameters(mean, std)
    return ModelParamsDataset(saved_trajectories, transform=normalizer if normalize else None), normalizer


def get_trajectory_dataloader(batch_size, models=None, pt_files=None, normalize=True, shuffle=True, device=None):
    dataset, normalizer = get_trajectory_dataset(models if models else pt_files, normalize=normalize)

    if models:
        # generator = torch.Generator(device=device)  # CUDA mps
        generator = torch.Generator(device='cpu')
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader, normalizer


def get_anchor_dataloader(dataset, subset=None):
    if subset is None:
        subset = range(len(dataset))
    dataset2 = torch.utils.data.Subset(dataset, subset)
    data_loader2 = DataLoader(dataset2, batch_size=len(dataset2), shuffle=False)
    print('number of constrained models considered: ', len(dataset2))

    return data_loader2


def get_predefined_values(dataset, anchor_mode="diagonal"):
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
