import torch.nn as nn
import numpy as np
import torch


# MODEL #

class Encoder(nn.Module):
    """
    A feedforward encoder network that maps the input to a latent space.
    
        Attributes:
          fcs (nn.Sequential): A sequential container holding the fully connected layers,
            layer normalizations, and ReLU activations of the encoder.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Initializes the Encoder.
        
                This method constructs the encoder network, which transforms the input data into a lower-dimensional latent space representation. This transformation is achieved through a sequence of fully connected layers, each followed by layer normalization and ReLU activation functions, designed to capture the essential features of the input data. The encoder is a crucial component for dimensionality reduction and feature extraction, enabling efficient processing and analysis of complex datasets.
        
                Args:
                    input_dim (int): The dimension of the input data.
                    hidden_dims (list[int]): A list of integers representing the dimensions of the hidden layers.
                    latent_dim (int): The dimension of the latent space.
        
                Returns:
                    None
        """
        super(Encoder, self).__init__()

        self.fcs = [nn.Linear(input_dim, hidden_dims[1]), nn.LayerNorm(hidden_dims[1]), nn.ReLU()]

        for i in range(1, len(hidden_dims)-2):
            self.fcs.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.fcs.append(nn.LayerNorm(hidden_dims[i+1]))
            self.fcs.append(nn.ReLU())
        self.fcs.append(nn.Linear(hidden_dims[-2], latent_dim))

        self.fcs = nn.Sequential(*self.fcs)

    def forward(self, x):
        """
        Performs a forward pass through the fully connected layers and applies a tanh activation. This step transforms the input into a latent representation suitable for approximating the solution of the differential equation.
        
                Args:
                    x (torch.Tensor): The input tensor.
        
                Returns:
                    torch.Tensor: The output tensor after passing through the fully connected layers and applying the tanh activation function, representing the encoded state.
        """
        x = self.fcs(x)
        z = torch.tanh(x)
        return z


class Decoder(nn.Module):
    """
    A decoder network constructed using fully connected layers.
    
        Class Methods:
        - __init__:
    """

    def __init__(self, latent_dim, hidden_dims, output_dim):
        """
        Initializes the Decoder module.
        
                This method constructs a decoder network using fully connected layers to map from the latent space back to the original data space.
                The network architecture is determined by the provided dimensions, allowing the model to reconstruct the original data from its encoded representation.
        
                Args:
                    latent_dim (int): The dimension of the latent space. This represents the encoded representation of the input data.
                    hidden_dims (list[int]): A list of integers representing the dimensions of the hidden layers. These layers help in learning complex mappings from the latent space.
                    output_dim (int): The dimension of the output layer. This should match the dimension of the original data being reconstructed.
        
                Returns:
                    None
        
                Class Fields:
                    fcs (nn.Sequential): A sequential container holding the fully connected layers, layer normalizations, and ReLU activations of the decoder network.
                    These layers transform the latent representation back into the original data space, approximating the inverse of the encoding process.
        """
        super(Decoder, self).__init__()

        self.fcs = [nn.Linear(latent_dim, hidden_dims[1]), nn.LayerNorm(hidden_dims[1]), nn.ReLU()]

        for i in range(1, len(hidden_dims)-2):
            self.fcs.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.fcs.append(nn.LayerNorm(hidden_dims[i+1]))
            self.fcs.append(nn.ReLU())
        self.fcs.append(nn.Linear(hidden_dims[-2], output_dim))

        self.fcs = nn.Sequential(*self.fcs)

    def forward(self, x):
        """
        Performs a forward pass through the fully connected layers.
        
        Applies the fully connected layers to the input and returns the result. This step is crucial for transforming the input into a suitable representation for approximating the solution of the differential equation.
        
        Args:
          x: The input tensor representing the encoded state.
        
        Returns:
          The output tensor after passing through the fully connected layers, representing the decoded state.
        """
        x = self.fcs(x)
        z = x
        return z


def get_hidden_layer_sizes(num_of_inputs, num_of_outputs, num_of_layers):
    """
    Generates a list of hidden layer sizes for a neural network-based differential equation solver.
    
        This method determines the architecture of the neural network by calculating the sizes of hidden layers.
        It uses a logarithmic scale based on the number of inputs, outputs, and desired layers to define the
        intermediate layer sizes, ensuring a smooth transition between input and output dimensions. This is
        crucial for the neural network to effectively learn the underlying function representing the solution
        of the differential equation.
    
        Args:
            num_of_inputs (int): The number of input features, corresponding to the initial conditions or
                                 independent variables of the differential equation.
            num_of_outputs (int): The number of output features, representing the predicted solution values
                                  of the differential equation.
            num_of_layers (int): The desired number of hidden layers in the neural network.
    
        Returns:
            list: A list of integers representing the sizes of each layer in the neural network, including
                  the input and output layers. These sizes define the structure of the network used to
                  approximate the solution of the differential equation.
    """
    if num_of_layers < 2 <= num_of_layers:
        raise ValueError("The number of layers must be at least 2.")
    if num_of_inputs < num_of_outputs:
        raise ValueError("Input size must be greater than or equal to the output size.")

    layer_sizes = np.logspace(np.log10(num_of_inputs), np.log10(num_of_outputs), num_of_layers + 2, dtype=int)
    return layer_sizes.tolist()


class UniformAutoencoder(nn.Module):
    """
    A uniform autoencoder model with configurable encoder and decoder networks.
    
        Class Methods:
        - __init__: Initializes a UniformAutoencoder.
        - forward: Performs a forward pass through the autoencoder.
    
        Class Fields:
        - encoder (Encoder): The encoder network.
        - decoder (Decoder): The decoder network.
    """

    def __init__(self, input_dim, num_of_layers, latent_dim, h=None):
        """
        Initializes a UniformAutoencoder, which is a core component for approximating solutions to differential equations using neural networks. It sets up the encoder and decoder networks that learn to compress and reconstruct the input data, effectively capturing the underlying dynamics of the differential equation.
        
                Args:
                    input_dim: The dimension of the input data, corresponding to the number of variables in the differential equation.
                    num_of_layers: The number of layers in the encoder/decoder, influencing the model's capacity to learn complex relationships.
                    latent_dim: The dimension of the latent space, representing the compressed representation of the input data.
                    h: Hidden layer sizes for the encoder. If None, it will be
                        calculated using `get_hidden_layer_sizes`.
        
                Returns:
                    None
        
                Class Fields:
                    encoder (Encoder): The encoder network, responsible for mapping the input data to the latent space.
                    decoder (Decoder): The decoder network, responsible for reconstructing the input data from the latent space.
        """
        super(UniformAutoencoder, self).__init__()
        if h is None:
            h = get_hidden_layer_sizes(input_dim, latent_dim, num_of_layers)
        self.encoder = Encoder(input_dim, h, latent_dim)
        self.decoder = Decoder(latent_dim, list(reversed(h)), input_dim)

    def forward(self, x):
        """
        Performs a forward pass to approximate the solution of a differential equation.
        
                Encodes the input using a neural network, then decodes the encoded representation to reconstruct the input, effectively learning the underlying dynamics of the differential equation.
        
                Args:
                    x (torch.Tensor): The input tensor representing the initial conditions or time points.
        
                Returns:
                    Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    - The reconstructed input tensor, representing the approximated solution.
                    - The encoded representation tensor, capturing the learned features.
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
