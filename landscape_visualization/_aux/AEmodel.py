import torch.nn as nn
import numpy as np
import torch

#####MODEL######

def get_hidden_layer_sizes(num_of_inputs, num_of_outputs, num_of_layers): 
    if num_of_layers < 2:
        raise ValueError("The number of layers must be at least 2.")
    if num_of_inputs < num_of_outputs:
        raise ValueError("Input size must be greater than or equal to the output size.")
    
    layer_sizes = np.logspace(np.log10(num_of_inputs), np.log10(num_of_outputs), num_of_layers+2, dtype=int)
    return layer_sizes.tolist()



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()

        self.fcs = [nn.Linear(input_dim, hidden_dims[1]), nn.LayerNorm(hidden_dims[1]), nn.ReLU()]

        for i in range(1, len(hidden_dims)-2):
            self.fcs.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.fcs.append(nn.LayerNorm(hidden_dims[i+1]))
            self.fcs.append(nn.ReLU())
        self.fcs.append(nn.Linear(hidden_dims[-2], latent_dim))

        self.fcs = nn.Sequential(*self.fcs)

    def forward(self, x):
        x = self.fcs(x)
        z = torch.tanh(x) 
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()

        self.fcs = [nn.Linear(latent_dim, hidden_dims[1]), nn.LayerNorm(hidden_dims[1]), nn.ReLU()]

        for i in range(1, len(hidden_dims)-2):
            self.fcs.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]) )
            self.fcs.append(nn.LayerNorm(hidden_dims[i+1]))
            self.fcs.append(nn.ReLU())
        self.fcs.append(nn.Linear(hidden_dims[-2], output_dim))

        self.fcs = nn.Sequential(*self.fcs)

    def forward(self, x):
        x = self.fcs(x)
        z=x
        return z
    



class UniformAutoencoder(nn.Module):
    def __init__(self, input_dim, num_of_layers, latent_dim, h=None):
        super(UniformAutoencoder, self).__init__()
        if h is None:
            h = get_hidden_layer_sizes(input_dim, latent_dim, num_of_layers)
        self.encoder = Encoder(input_dim, h, latent_dim)
        self.decoder = Decoder(latent_dim, list(reversed(h)), input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
