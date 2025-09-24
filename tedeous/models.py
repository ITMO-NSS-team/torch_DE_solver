"""Module keeps custom models arctectures"""
from typing import List, Any
import torch
from torch import nn
import numpy as np


class Fourier_embedding(nn.Module):
    """
    Class for generating Fourier feature embeddings, useful for encoding input data into a higher-dimensional space to improve model performance.
    
    
        Examples:
            u(t,x) if user wants to create 5 Fourier features in 'x' direction with L=5:
                L=[None, 5], M=[None, 5].
    """


    def __init__(self, L=[1], M=[1], ones=False):
        """
        Initializes the Fourier embedding layer. This layer transforms the input by projecting it onto a higher-dimensional space using Fourier features, which can help the neural network to better approximate the solution of the differential equation.
        
                Args:
                    L (list, optional): Characteristic length scales used to define the frequencies for the Fourier basis functions.  Each element corresponds to a different frequency component. Defaults to [1].
                    M (list, optional): Number of sine and cosine pairs to use for each frequency specified in `L`.  Determines the dimensionality of the Fourier features. Defaults to [1].
                    ones (bool, optional): Whether to include a vector of ones in the output embedding. This can provide a bias term for the subsequent layers. Defaults to False.
        
                Returns:
                    None
        """

        super().__init__()
        self.M = M
        self.L = L
        self.idx = [i for i in range(len(self.M)) if self.M[i] is None]
        self.ones = ones
        self.in_features = len(M)
        not_none = sum(i for i in M if i is not None)
        is_none = self.M.count(None)
        if is_none == 0:
            self.out_features = not_none * 2 + self.in_features
        else:
            self.out_features = not_none * 2 + is_none
        if ones is not False:
            self.out_features += 1

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input grid into a higher-dimensional space using Fourier features.
        
        This transformation enhances the representation of the input,
        allowing a neural network to better capture the underlying patterns
        when solving differential equations. By mapping the grid to Fourier features,
        the network can learn complex relationships more effectively.
        
        Args:
            grid (torch.Tensor): The calculation domain, representing the independent variable(s)
                                  over which the differential equation is defined.
        
        Returns:
            torch.Tensor: The embedding of the grid with Fourier features. This enriched
                          representation is then used as input to the neural network.
        """

        if self.idx == []:
            out = grid
        else:
            out = grid[:, self.idx]

        for i, _ in enumerate(self.M):
            if self.M[i] is not None:
                Mi = self.M[i]
                Li = self.L[i]
                w = 2.0 * np.pi / Li
                k = torch.arange(1, Mi + 1).reshape(-1, 1).float()
                x = grid[:, i].reshape(1, -1)
                x = (k @ x).T
                embed_cos = torch.cos(w * x)
                embed_sin = torch.sin(w * x)
                out = torch.hstack((out, embed_cos, embed_sin))

        if self.ones is not False:
            out = torch.hstack((out, torch.ones_like(out[:, 0:1])))

        return out


class FourierNN(nn.Module):
    """
    Class for realizing neural network with Fourier features
        and skip connection.
    """


    def __init__(self, layers=[100, 100, 100, 1], L=[1], M=[1],
                 activation=nn.Tanh(), ones=False):
        """
        Initializes the Fourier Neural Network (FourierNN) model.
        
                This method sets up the architecture of the neural network, incorporating a Fourier embedding layer to map the input to a higher-dimensional space using sinusoidal functions. This embedding helps the network to better capture the underlying patterns in the data, which is particularly useful when solving differential equations. The network consists of a sequence of linear layers with an activation function applied between them.
        
                Args:
                    layers (list, optional): Number of neurons in each layer (excluding the input layer). The number of neurons in the hidden layers must match. Defaults to [100, 100, 100, 1].
                    L (list, optional): Frequency parameter for the Fourier embedding, where w = 2*pi/L. Defaults to [1].
                    M (list, optional): Number of (sin, cos) pairs in the Fourier embedding. Defaults to [1].
                    activation (nn.Module, optional): Activation function to be applied between the linear layers. Defaults to nn.Tanh().
                    ones (bool, optional): Whether to include a vector of ones in the Fourier embedding. Defaults to False.
        
                Returns:
                    None
        """

        super(FourierNN, self).__init__()
        self.L = L
        self.M = M
        FFL = Fourier_embedding(L=L, M=M, ones=ones)

        layers = [FFL.out_features] + layers

        self.linear_u = nn.Linear(layers[0], layers[1])
        self.linear_v = nn.Linear(layers[0], layers[1])

        self.activation = activation
        self.model = nn.ModuleList([FFL])

        for i in range(len(layers) - 1):
            self.model.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Applies the neural network to approximate the solution of a differential equation on a given domain.
        
        The network processes the input grid through a series of layers,
        refining the approximation at each step. The initial input is transformed
        and then passed through a series of layers with skip connections,
        allowing the network to learn complex relationships within the data
        and refine the solution iteratively.
        
        Args:
            grid (torch.Tensor): The spatial or temporal domain over which the differential equation is defined.
        
        Returns:
            torch.Tensor: The predicted solution of the differential equation at each point on the input grid.
        """

        input_ = self.model[0](grid)
        v = self.activation(self.linear_v(input_))
        u = self.activation(self.linear_u(input_))
        for layer in self.model[1:-1]:
            output = self.activation(layer(input_))
            input_ = output * u + (1 - output) * v

        output = self.model[-1](input_)

        return output


class FeedForward(nn.Module):
    """
    Simple MLP neural network
    """


    def __init__(self,
                 layers: List = [2, 100, 100, 100, 1],
                 activation: nn.Module = nn.Tanh(),
                 parameters: dict = None):
        """
        Initializes the FeedForward neural network.
        
                This method sets up the architecture of the neural network, defining the layers,
                activation functions, and optionally initializing the network's parameters.
                The network is constructed as a sequence of linear layers with specified activation functions
                in between, designed to approximate solutions to differential equations.
        
                Args:
                    layers (List, optional): A list specifying the number of neurons in each layer of the network.
                        Defaults to [2, 100, 100, 100, 1], representing an input layer with 2 neurons, three hidden
                        layers with 100 neurons each, and an output layer with 1 neuron.
                    activation (nn.Module, optional): The activation function to be applied between the linear layers.
                        Defaults to nn.Tanh().
                    parameters (dict, optional): A dictionary containing initial values for the network's parameters.
                        If provided, these values are used to initialize the network's weights and biases,
                        potentially guiding the solution towards a desired or known state. Defaults to None.
        
                Returns:
                    None
        """

        super().__init__()
        self.model = []

        for i in range(len(layers) - 2):
            self.model.append(nn.Linear(layers[i], layers[i + 1]))
            self.model.append(activation)
        self.model.append(nn.Linear(layers[-2], layers[-1]))
        self.net = torch.nn.Sequential(*self.model)
        if parameters is not None:
            self.reg_param(parameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feedforward network to the input tensor.
        
        This method propagates the input `x` through the defined neural network (`self.net`) to approximate the solution of a differential equation.
        
        Args:
            x (torch.Tensor): The input tensor representing the independent variable(s) of the differential equation.
        
        Returns:
            torch.Tensor: The output tensor representing the approximated solution of the differential equation at the input points.
        """
        return self.net(x)

    def reg_param(self,
                  parameters: dict):
        """
        Registers provided parameters as learnable `torch.nn.Parameter` within the neural network.
        
        This is crucial for tasks where specific coefficients or parameters within the differential equation
        need to be optimized alongside the neural network's weights. By registering these parameters,
        they become part of the optimization process, allowing the network to learn the optimal values
        for these parameters in order to better approximate the solution of the differential equation.
        
        Args:
            parameters (dict): A dictionary where keys are parameter names (strings) and values are their initial numerical values.
        
        Returns:
            None. The method modifies the internal state of the neural network (`self.net`) by registering the parameters.
        """
        for key, value in parameters.items():
            parameters[key] = torch.nn.Parameter(torch.tensor([value],
                                                              requires_grad=True).float())
            self.net.register_parameter(key, parameters[key])


def parameter_registr(model: torch.nn.Module,
                      parameters: dict) -> None:
    """
    Registers given parameters as learnable parameters of the neural network.
    
    This allows the optimization process to adjust these parameters during the training 
    to better approximate the solution of the differential equation.
    
    Args:
        model (torch.nn.Module): The neural network model.
        parameters (dict): A dictionary containing the initial values of the parameters.
    
    Returns:
        None
    """
    for key, value in parameters.items():
        parameters[key] = torch.nn.Parameter(torch.tensor([value],
                                                          requires_grad=True).float())
        model.register_parameter(key, parameters[key])


def mat_model(domain: Any,
              equation: Any,
              nn_model: torch.nn.Module = None) -> torch.Tensor:
    """
    Creates a model that represents the solution of a differential equation on a grid.
    
    This function constructs a solution model by either utilizing a provided neural network
    or initializing a tensor of ones. The model represents the approximate solution
    of the differential equation defined by the equation object, evaluated on a grid
    generated from the domain.
    
    Args:
        domain (Any): An object representing the domain over which the differential equation is defined.
            It is used to build the grid where the solution will be evaluated.
        equation (Any): An object containing the differential equation(s) to be solved.
            The number of equations determines the first dimension of the output model.
        nn_model (torch.nn.Module, optional): A neural network model that approximates the solution.
            If provided, the grid is passed through the network to obtain the solution.
            If None, a tensor of ones is used as the initial guess. Defaults to None.
    
    Returns:
        torch.Tensor: A tensor representing the approximate solution of the differential equation
        on the defined grid. If a neural network is provided, this is the network's output;
        otherwise, it's a tensor of ones.
    """

    grid = domain.build('mat')

    eq_num = len(equation.equation_lst)

    shape = [eq_num] + list(grid.shape)[1:]

    if nn_model is not None:
        nn_grid = torch.vstack([grid[i].reshape(-1) for i in range(grid.shape[0])]).T.float()
        model = nn_model(nn_grid).detach()
        model = model.reshape(shape)
    else:
        model = torch.ones(shape)

    return model
