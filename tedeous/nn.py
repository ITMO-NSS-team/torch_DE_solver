import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_sizes, output_layer_size, activations):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.activations = activations
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        in_size = input_layer_size
        for i, out_size in enumerate(hidden_layer_sizes):
            self.hidden_layers.append(nn.Linear(in_size, out_size))
            in_size = out_size

        self.output_layer = nn.Linear(in_size, output_layer_size)

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.activations == 'relu':
                x = self.relu(x)
            elif self.activations == 'tanh':
                x = self.tanh(x)
            else:
                raise ValueError(f"Invalid activation function: {self.activations}")

        x = self.output_layer(x)
        # print(x.shape)
        return x




class FourierFeatureNetwork(NN):
    def __init__(self, input_layer_size, hidden_layer_sizes, output_layer_size, activations, sigma):
        super().__init__(input_layer_size, hidden_layer_sizes, output_layer_size, activations)
        self.sigma = sigma
        self.input_layer_size = input_layer_size
        self.hidden_layer_sizes = hidden_layer_sizes

    def get_fourier_features(self, x, sigma):
        # print(self.input_layer_size)
        y = sigma * torch.normal(mean = 0, std = 1, size = [1, self.hidden_layer_sizes[0] // 2])
        # y = sigma * torch.normal(mean=0., std=1., size=[2,2])
        print(y.shape)
        print(x.shape)
        z = torch.matmul(x,y)
        real = torch.cos(torch.mm(x, y))
        imag = torch.sin(torch.mm(x, y))
        # print(real.shape)

        fourier_features = torch.cat((real, imag), dim=1)
        # print(real.shape)
        return fourier_features, y

    def forward(self, x):
        x, _ = self.get_fourier_features(x,self.sigma)
        # print(x.shape)
        x = super().forward(x)
        x = torch.cat(x, 1)
        return super().forward(x)
