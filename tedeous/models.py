"""Module keeps custom models arctectures"""

from typing import List, Any
import torch
import math
from torch import nn
import numpy as np
import torch.nn.functional as F


class Fourier_embedding(nn.Module):
    """
    Class for Fourier features generation.

    Examples:
        u(t,x) if user wants to create 5 Fourier features in 'x' direction with L=5:
            L=[None, 5], M=[None, 5].
    """

    def __init__(self, L=[1], M=[1], ones=False):
        """
        Args:
            L (list, optional): (sin(w*x), cos(w*X)) frequencie parameter,
            w = 2*pi/L. Defaults to [1].
            M (list, optional): number of (sin, cos) pairs in result embedding. Defaults to [1].
            ones (bool, optional): enter or not ones vector in result embedding. Defaults to False.
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
        """ Forward method for Fourier features generation.

        Args:
            grid (torch.Tensor): calculation domain.

        Returns:
            torch.Tensor: embedding with Fourier features.
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
        Args:
            layers (list, optional): neurons quantity in each layer (exclusion input layer),
            the number of neurons in the hidden layers must match. Defaults to [100, 100, 100, 1].
            L (list, optional): (sin(w*x),cos(w*x)) frequency parameter, w=2*pi/L. Defaults to [1].
            M (list, optional): number of (sin, cos) pairs in result embedding. Defaults to [1].
            activation (_type_, optional): nn.Module object, activ-n function. Defaults to nn.Tanh().
            ones (bool, optional): enter or not ones vector in result embedding. Defaults to False.
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
        """ Forward pass for neural network.

        Args:
            grid (torch.Tensor): calculation domain.

        Returns:
            torch.Tensor: predicted values.
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
    """Simple MLP neural network"""

    def __init__(self,
                 layers: List = [2, 100, 100, 100, 1],
                 activation: nn.Module = nn.Tanh(),
                 parameters: dict = None):
        """
        Args:
            layers (List, optional): neurons quantity in each layer.
            Defaults to [2, 100, 100, 100, 1].
            activation (nn.Module, optional): nn.Module object, activ-n function.
            Defaults to nn.Tanh().
            parameters (dict, optional): parameters initial values (for inverse task).
            Defaults to None.
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
        """ forward run

        Args:
            x (torch.Tensor): neural network inputs

        Returns:
            torch.Tensor: outputs
        """
        return self.net(x)

    def reg_param(self,
                  parameters: dict):
        """ Parameters registration as neural network parameters.
        Should be used in inverse coefficients tasks.

        Args:
            parameters (dict): dict with initial values.
        """
        for key, value in parameters.items():
            parameters[key] = torch.nn.Parameter(torch.tensor([value],
                                                              requires_grad=True).float())
            self.net.register_parameter(key, parameters[key])


class KANLinear(torch.nn.Module):
    """
    Class that implements linear layer using Kolmogorov-Arnold splines.
    It allows you to model nonlinear dependencies between input and output data using splines.
    """

    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        """
        A method for initializing layer parameters,
        including base model weights and spline weights.
        """

        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self,
                  x: torch.Tensor
                  ) -> torch.Tensor:
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self,
                    x: torch.Tensor,
                    y: torch.Tensor
                    ) -> torch.Tensor:
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(
            A, B
        ).solution
        result = solution.permute(
            2, 0, 1
        )

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self,
                    x: torch.Tensor,
                    margin=0.01):
        """ Method to update the grid based on the input data to improve interpolation

            Args:
                x (torch.tensor): Input tensor of shape (batch_size, in_features).
                margin: The value to be added to the grid value range. Defaults to 0.01
        """

        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self,
                            regularize_activation=1.0,
                            regularize_entropy=1.0):
        """ Calculate regularization loss for spline weights

            Args:
                regularize_activation (float): Regularization factor for activation. Defaults to 1.0.
                regularize_entropy (float): Regularization factor for entropy. Defaults to 1.0.

            Returns:
                torch.Tensor: Regularization loss
        """

        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())

        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    """Class for realization of multilayer network using Kolmogorov-Arnold splines."""

    def __init__(
            self,
            layers_hidden=[2, 100, 1],
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.model = nn.ModuleList()

        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.model.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self,
                x: torch.Tensor,
                update_grid=False
                ) -> torch.Tensor:
        """ Forward run

            Args:
                x (torch.Tensor): Network inputs.
                update_grid (bool): Flag to update the grid in each layer. Defaults to False.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, out_features).
        """

        for layer in self.model:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)

        return x

    def regularization_loss(self,
                            regularize_activation=1.0,
                            regularize_entropy=1.0):
        """ Calculate regularization loss for all network

            Args:
                regularize_activation (float): Regularization factor for activation. Defaults to 1.0.
                regularize_entropy (float): Regularization factor for entropy. Defaults to 1.0.

            Returns:
                Regularization loss
        """

        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.model
        )


def parameter_registr(model: torch.nn.Module,
                      parameters: dict) -> None:
    """Parameters registration as neural network (mpdel) parameters.
        Should be used in inverse coefficients tasks.

    Args:
        model (torch.nn.Module): neural network.
        parameters (dict): dict with initial values.
    """
    for key, value in parameters.items():
        parameters[key] = torch.nn.Parameter(torch.tensor([value],
                                                          requires_grad=True).float())
        model.register_parameter(key, parameters[key])


def mat_model(domain: Any,
              equation: Any,
              nn_model: torch.nn.Module = None) -> torch.Tensor:
    """ Model creation for *mat* mode.

    Args:
        domain (Any): object of Domian class.
        equation (Any): Equation class object (see data module).
        nn_model (torch.nn.Module, optional): neural network which outputs will be *mat* model.
        Defaults to None.

    Returns:
        torch.nn.Module: model for *mat* mode.
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
