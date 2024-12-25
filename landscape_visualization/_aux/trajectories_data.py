# This code is partially based on the repository source: https://github.com/elhamod/NeuroVisualizer.git.
from torch.utils.data import DataLoader, Dataset
import torch
import builtins
import numpy as np
torch.serialization.add_safe_globals([torch.nn.Sequential,torch.nn.modules.linear.Linear,torch.nn.modules.activation.Tanh, builtins.set])


def calculate_mean_std(file_paths, path):
    state_dicts = [torch.load(file_path, map_location=torch.device('cpu'), weights_only = True) for file_path in file_paths]
    isSpecialCase = not hasattr(state_dicts[0], 'keys')
    keys = list(state_dicts[0].keys() if not isSpecialCase else state_dicts[0].state_dict().keys())
    mean_values = []
    std_values = []
    for key in keys:
        if not isSpecialCase:
            values = [state_dict[key].float().view(1, -1) for state_dict in state_dicts]
        else:
            values = [state_dict.state_dict()[key].float().view(1, -1) for state_dict in state_dicts]

        mean = torch.mean(torch.stack(values), dim=0)
        std = torch.std(torch.stack(values), dim=0)

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
    def __init__(self, file_paths, transform=None, model_dict=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        model_dict = torch.load(file_path, map_location=torch.device('cpu'), weights_only = True)
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
    

def get_trajectory_dataset(pt_files, path, normalize=True, model_dict=None):
    # print('pt_files', pt_files, path)
    mean, std = calculate_mean_std(pt_files, path)
    normalizer = NormalizeModelParameters(mean, std)
    return ModelParamsDataset(pt_files, transform=normalizer if normalize else None, model_dict=model_dict), normalizer

def get_trajectory_dataloader(pt_files, batch_size, path, normalize=True, shuffle=True, model_dict=None):
    dataset, normalizer = get_trajectory_dataset(pt_files, path, normalize=normalize, model_dict=model_dict)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader, normalizer



def get_anchor_dataloader(dataset, subset=None):
    if subset is None:
        subset= range(len(dataset))
    dataset2 = torch.utils.data.Subset(dataset, subset)
    data_loader2 = DataLoader(dataset2, batch_size=len(dataset2), shuffle=False)
    print('number of constrained models considered: ', len(dataset2))

    return data_loader2

def get_predefined_values(dataset, anchor_mode="diagonal"):
    if anchor_mode=="diagonal":
        # Predefined set of values
        begin = -1.0  # endpoint of the array
        end = 1.0
        s = len(dataset) # number of steps between start and endpoint

        # Create an array of s+2 points with [0.0, 0.0] as the first point and [n, n] as the last point
        points = np.linspace(begin, end, s)
        # Create an array of tuples from the points array
        tuples_array = np.array([(x, y) for x in points for y in points])
        # Filter the tuples array to only include tuples within the range of [0.0, 0.0] and [n, n]
        tuples_array = tuples_array[(tuples_array[:, 0] <= end) & (tuples_array[:, 1] <= end) & (tuples_array[:, 0] == tuples_array[:, 1])]
        predefined_values = torch.tensor(tuples_array, dtype=torch.float32)
    elif anchor_mode=="circle":
        # Example data
        n = len(dataset)  # Number of points on the circle
        r = 0.8 # Radius of the circle

        # Generate coordinates
        theta = torch.linspace(0, 2*torch.pi, n+1)[:-1]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        predefined_values = torch.stack([x, y], dim=1)

    else:
        raise "anchor_mode not implemented"
    return predefined_values