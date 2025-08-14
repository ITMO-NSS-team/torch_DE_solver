import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

class DQN_optim(nn.Module):
    def __init__(self, optim_n):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(p=0.1)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)   # → (B,64,1,1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(),
        )
        self.fc_optim_class = nn.Linear(64, optim_n)
        # self.softmax  = nn.Softmax(dim=1)

        for m in self.modules():                 # He init
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.backbone(x)
        flat = self.gap(x).view(x.size(0), -1)
        h    = self.head(flat)
        return flat, self.fc_optim_class(h)
    

# class DQN_params(nn.Module):
#     def __init__(self, optimizer_dict):
#         super(DQN_params, self).__init__()
#         self.optimizer_dict = optimizer_dict
#         layers_ar = []
#         fc_liner = lambda param_var: (nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
#                                 nn.Linear(128, 64), nn.ReLU(), 
#                                 nn.Linear(64, len(param_var)))
#         self.fc_param_by_opt = defaultdict(defaultdict)
#         for opt_name in self.optimizer_dict.keys():
#             for param_name in self.optimizer_dict[opt_name].keys():
#                 param_var = self.optimizer_dict[opt_name][param_name]
#                 # self.fc_param_by_opt[opt_name][param_name] = nn.Linear(128, len(param_var))
#                 linear_layer = fc_liner(param_var)
#                 self.fc_param_by_opt[opt_name][param_name] = linear_layer
#                 layers_ar += list(linear_layer)
#         self.linears = nn.ModuleList(layers_ar)

#     def forward(self, x, optim_name_ar):
#         x_params_ar = []
#         for i, optim_name in enumerate(optim_name_ar):
#             x_params = {}
#             for param in self.fc_param_by_opt[optim_name].keys():
#                 param_liner = self.fc_param_by_opt[optim_name][param]
#                 x_ = x[i]
#                 for fc_lin in param_liner:
#                     x_ = fc_lin(x_)
#                 x_params[param] = x_
#             x_params_ar.append(x_params)
#         return x_params_ar
    

class DQN_params(nn.Module):
    def __init__(self, optimizer_dict):
        super().__init__()
        self.optimizer_dict = optimizer_dict
        self.fc_param_by_opt = nn.ModuleDict()

        for opt_name, param_dict in optimizer_dict.items():
            self.fc_param_by_opt[opt_name] = nn.ModuleDict()
            for param_name, param_values in param_dict.items():
                self.fc_param_by_opt[opt_name][param_name] = nn.Sequential(
                    nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Linear(64, len(param_values))
                )

    def forward(self, x, optim_name_ar):
        x_params_ar = []
        for i, optim_name in enumerate(optim_name_ar):
            x_params = {}
            for param_name, net in self.fc_param_by_opt[optim_name].items():
                x_ = x[i]
                x_params[param_name] = net(x_)
            x_params_ar.append(x_params)
        return x_params_ar
