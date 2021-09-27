import torch
import numpy as np


def derivative(u_tensor, h_tensor, axis, scheme_order=1, boundary_order=1):

    u_tensor = torch.transpose(u_tensor, 0, axis)
    h_tensor = torch.transpose(h_tensor, 0, axis)

    du_forward = (-torch.roll(u_tensor, -1) + u_tensor) / \
                 (-torch.roll(h_tensor, -1) + h_tensor)
    du_backward = (torch.roll(u_tensor, 1) - u_tensor) / \
                  (torch.roll(h_tensor, 1) - h_tensor)

    du = (1/2) * (du_forward + du_backward)

    # ind = torch.zeros(du.shape, dtype=torch.long, device=device)
    # values = torch.gather(du_forward, 1, ind)
    du[:, 0] = du_forward[:, 0]
    # du = du.scatter_(1, ind, values)

    # ind = (du.shape[axis] - 1) * torch.ones(du.shape, dtype=torch.long, device=device)
    # values = torch.gather(du_backward, 1, ind)
    # du = du.scatter_(1, ind, values)
    du[:, -1] = du_backward[:, -1]

    du = torch.transpose(du, 0, axis)

    return du
