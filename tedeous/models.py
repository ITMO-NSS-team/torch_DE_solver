import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Define MLP with exact periodicity 
class MLP(nn.Module):
    def __init__(self, layers=[100,100,100,1], L=[1], M=[1],
                 activation=F.tanh, device='cpu'):
        super(MLP, self).__init__()
        self.L = L
        self.M = M
        self.device = device
        not_none = sum([i for i in M if i is not None])
        is_none = self.M.count(None)
        d0 = not_none*2 + is_none + 1
        layers = [d0] + layers
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1], device=device
                      ) for i in range(len(layers)-1)])
        self.activation = activation

    def input_encoding(self, grid):
        idx = [i for i in range(len(self.M)) if self.M[i] is None]
        features = grid[:,idx]
        ones = torch.ones_like(grid[:,0:1]).to(self.device)
        for i in range(len(self.M)):
            if self.M[i] is not None:
                Mi = self.M[i]
                Li = self.L[i]
                w = 2.0 * np.pi / Li
                k = torch.arange(1, Mi+1).to(self.device).reshape(-1,1).float()
                x = grid[:,i].reshape(1,-1)
                x = (k@x).T
                embed_cos = torch.cos(w*x)
                embed_sin = torch.sin(w*x)
        
        out = torch.hstack((embed_cos, embed_sin, ones, features))
        return out

    def forward(self, grid):
        H = self.input_encoding(grid)
        for layer in self.linears[:-1]:
            H = self.activation(layer(H))
        outputs = self.linears[-1](H)
        return outputs


# Define modified MLP
class Modified_MLP(nn.Module):
    def __init__(self, layers=[100,100,100,1], L=[1], M=[1],
                 activation=F.tanh, device='cpu'):
        super(Modified_MLP, self).__init__()
        self.L = L
        self.M = M
        self.device = device
        not_none = sum([i for i in M if i is not None]) 
        is_none = self.M.count(None)
        if is_none == 0:
            layers = [not_none*2 + 1] + layers
        else:
            layers = [not_none*2 + is_none] + layers
        self.layers = layers
        self.linear_u = nn.Linear(layers[0], layers[1], device=device)
        self.linear_v = nn.Linear(layers[0], layers[1], device=device)
        self.linears = nn.ModuleList([
                    nn.Linear(layers[i], layers[i+1], device=device)
                                        for i in range(len(layers)-2)])
        self.linear_last = nn.Linear(layers[-2], layers[-1], device=device)
        self.activation = activation


    # Define input encoding function
    def input_encoding(self, grid):
        idx = [i for i in range(len(self.M)) if self.M[i] is None]
        if idx == []:
            out = grid
        else:
            out = grid[:,idx]
        
        for i in range(len(self.M)):
            if self.M[i] is not None:
                Mi = self.M[i]
                Li = self.L[i]
                w = 2.0 * np.pi / Li
                k = torch.arange(1, Mi+1).to(self.device).reshape(-1,1).float()
                x = grid[:,i].reshape(1,-1)
                x = (k@x).T
                embed_cos = torch.cos(w*x)
                embed_sin = torch.sin(w*x)
                out = torch.hstack((out, embed_cos, embed_sin))
        return out


    def forward(self, grid):   
        inputs = self.input_encoding(grid)
        U = self.activation(self.linear_u(inputs))
        V = self.activation(self.linear_v(inputs))
        for layer in self.linears:
            outputs = F.tanh(layer(inputs))
            inputs = outputs*U + (1 - outputs)*V

        outputs = self.linear_last(inputs)

        return outputs


class FirstLinear(nn.Linear): 
    def __init__(self, in_features, out_features, bias=True,
                                            device=None, dtype=None, sigma=1.):
        super(FirstLinear, self).__init__(in_features, out_features,
                                          bias, device, dtype)
        self.sigma = torch.tensor([sigma]).to('cuda')

    def forward(self, input):
        return F.linear(input, self.sigma * self.weight, self.bias) 


class FFN(nn.Module):
    def __init__(self, layers):
        super(FFN, self).__init__()
        self.layers = layers
        self.firstLinear1 = FirstLinear(1, layers[0] // 2, bias=False, sigma=1., device='cuda')
        self.firstLinear2 = FirstLinear(1, layers[0] // 2, bias=False, sigma=10., device='cuda')
        self.layerlist = nn.ModuleList([nn.Linear(layers[i], layers[i+1])
                                                for i in range(len(layers)-2)])
        self.lastLayer = nn.Linear(2*layers[-2], layers[-1])

    def forward(self, X):
        H1 = torch.cat((torch.sin(self.firstLinear1(X)),
                        torch.cos(self.firstLinear1(X))), dim=1)
        H2 = torch.cat((torch.sin(self.firstLinear2(X)),
                        torch.cos(self.firstLinear2(X))), dim=1)
        
        for layer in self.layerlist:
            H1 = F.tanh(layer(H1))
            H2 = F.tanh(layer(H2))

        H = torch.cat((H1, H2), dim=1)

        H = self.lastLayer(H)

        return H