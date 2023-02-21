import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_sizes, output_layer_size, activations,
                 fourier_features = False, sigma = None, mapping_size = None):
        super().__init__()
        self.input_layer_size = input_layer_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_layer_size = output_layer_size
        self.activations = activations
        self.fourier_features = fourier_features
        self.sigma = sigma
        self.mapping_size = mapping_size

        if self.activations == 'relu':
            self.activations = nn.ReLU()
        elif self.activations == 'tanh':
            self.activations = nn.Tanh()

        if self.fourier_features:
            self.input_layer_size = 2 * self.mapping_size

        self.model = []
        self.model.append(nn.Linear(self.input_layer_size, self.hidden_layer_sizes[0]))
        self.model.append(self.activations)
        for i in range(len(self.hidden_layer_sizes)):
            self.model.append(nn.Linear(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i]))
            self.model.append(self.activations)
        self.model.append(nn.Linear(self.hidden_layer_sizes[-1], self.output_layer_size))
        self.model = nn.Sequential(*self.model)

    def fourier_feature_mapping(self, x):
        mapping_size = (self.mapping_size, x.shape[-1])
        b = self.sigma * torch.normal(mean = 0, std = 1, size = mapping_size)
        x_proj = (2.*np.pi*x) @ b.T
        fourier_features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return fourier_features

    def forward(self,x):
        if self.fourier_features:
            x = self.fourier_feature_mapping(x)
        x = self.model(x)
        return x
