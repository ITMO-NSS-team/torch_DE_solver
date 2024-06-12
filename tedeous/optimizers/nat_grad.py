from typing import Tuple
import torch
import jax.numpy as jnp
from jax.numpy.linalg import lstsq
from copy import copy
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tedeous.device import device_type
from tedeous.utils import replace_none_by_zero

def grid_line_search_factory(loss, steps):

    def loss_at_step(step, model, tangent_params):
        params = parameters_to_vector(model.parameters())
        new_params = params - step*tangent_params
        vector_to_parameters(new_params, model.parameters())
        loss_val = loss(model, x_Omega, x_Gamma)
        vector_to_parameters(params, model.parameters())
        return loss_val


    def grid_line_search_update(model, tangent_params):

        losses = []
        for step in steps:
            losses.append(loss_at_step(step, model, tangent_params).reshape(1))
        losses = torch.cat(losses)
        step_size = steps[torch.argmin(losses)]

        params = parameters_to_vector(model.parameters())
        new_params = params - step_size*tangent_params
        vector_to_parameters(new_params, model.parameters())

        return step_size

    return grid_line_search_update


class NGD(torch.optim.Optimizer):
    def __init__(self, params):
        self.params = params
    
    def gram_factory(self, residuals):
        def jacobian():
            jac = []
            for l in residuals:
                j = torch.autograd.grad(l, self.params(), retain_graph=True, allow_unused=True)
                j = replace_none_by_zero(j)
                j = parameters_to_vector(j).reshape(1, -1)
                jac.append(j)
            return torch.cat(jac)

        J = jacobian()
        return 1.0 / len(residuals) * J.T @ J


    def step(self, closure=None):
        op_res, bval, true_bval, loss = closure()
        grads = torch.autograd.grad(loss, self.params(), retain_graph=True, allow_unused=True)
        grads = replace_none_by_zero(grads)
        f_grads = parameters_to_vector(grads)

        # assemble gramian
        G_int  = self.gram_factory(int_res)

        G_bdry = self.gram_factory(bound_res)
        G      = G_int + G_bdry

        # Marquardt-Levenberg
        Id = torch.eye(len(G))
        G = torch.min(torch.tensor([loss, 0.0])) * Id + G
        # compute natural gradient
        G = jnp.array(G.detach().cpu().numpy(), dtype=jnp.float64)
        f_grads =jnp.array(f_grads.detach().cpu().numpy(), dtype=jnp.float64)
        f_nat_grad = lstsq(G, f_grads)[0]
        f_nat_grad = torch.from_numpy(np.array(f_nat_grad)).to(torch.float64).to('cuda')

        # one step of NGD
        actual_step = ls_update(model, f_nat_grad)
