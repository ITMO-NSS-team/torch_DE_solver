import torch

def set_device(*args):
    if len(args) == 0:
        if torch.has_cuda:
            return torch.device('cuda')
        elif torch.has_mps:
            return torch.device('mps')
        else:
            return torch.device('cpu')
    if len(args) == 1:
        global device
        device = args[0]
        return torch.device(device)