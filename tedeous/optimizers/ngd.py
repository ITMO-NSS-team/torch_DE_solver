import torch
from numpy.linalg import lstsq
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tedeous.utils import replace_none_by_zero


class NGD(torch.optim.Optimizer):

    """NGD implementation (https://arxiv.org/abs/2302.13163).
    """

    def __init__(self, params,
                 grid_steps_number: int = 30):
        """The Natural Gradient Descent class.

        Args:
            grid_steps_number (int, optional): Grid steps number. Defaults to 30.
        """
        defaults = {'grid_steps_number': grid_steps_number}
        super(NGD, self).__init__(params, defaults)
        self.params = self.param_groups[0]['params']
        self.grid_steps_number = grid_steps_number
        self.grid_steps = torch.linspace(0, self.grid_steps_number, self.grid_steps_number + 1)
        self.steps = 0.5**self.grid_steps

    def grid_line_search_update(self, loss_function: callable, f_nat_grad: torch.Tensor) -> None:
        """ Update models paramters by natural gradient.

        Args:
            loss (callable): function to calculate loss.

        Returns:
            None.
        """
        # function to update models paramters at each step
        def loss_at_step(step, loss_function: callable, f_nat_grad: torch.Tensor) -> torch.Tensor:
            params = parameters_to_vector(self.params)
            new_params = params - step * f_nat_grad
            vector_to_parameters(new_params, self.params)
            loss_val, _ = loss_function()
            vector_to_parameters(params, self.params)
            return loss_val

        losses = []
        for step in self.steps:
            losses.append(loss_at_step(step, loss_function, f_nat_grad).reshape(1))
        losses = torch.cat(losses)
        step_size = self.steps[torch.argmin(losses)]

        params = parameters_to_vector(self.params)
        new_params = params - step_size * f_nat_grad
        vector_to_parameters(new_params, self.params)
    
    def gram_factory(self, residuals: torch.Tensor) -> torch.Tensor:
        """ Make Gram matrice.

        Args:
            residuals (callable): PDE residual.

        Returns:
            torch.Tensor: Gram matrice.
        """
        # Make Gram matrice.
        def jacobian() -> torch.Tensor:
            jac = []
            for l in residuals:
                j = torch.autograd.grad(l, self.params, retain_graph=True, allow_unused=True)
                j = replace_none_by_zero(j)
                j = parameters_to_vector(j).reshape(1, -1)
                jac.append(j)
            return torch.cat(jac)

        J = jacobian()
        return 1.0 / len(residuals) * J.T @ J

    def step(self, closure=None) -> torch.Tensor:
        """ It runs ONE step on the natural gradient descent.

        Returns:
            torch.Tensor: loss value for NGD step.
        """

        int_res, bval, true_bval, loss, loss_function = closure()
        grads = torch.autograd.grad(loss, self.params, retain_graph=True, allow_unused=True)
        grads = replace_none_by_zero(grads)
        f_grads = parameters_to_vector(grads)

        bound_res = bval-true_bval

        # assemble gramian
        G_int  = self.gram_factory(int_res)
        G_bdry = self.gram_factory(bound_res)
        G      = G_int + G_bdry

        # Marquardt-Levenberg
        Id = torch.eye(len(G))
        G = torch.min(torch.tensor([loss, 0.0])) * Id + G

        # compute natural gradient
        G = np.array(G.detach().cpu().numpy(), dtype=np.float32)
        f_grads = np.array(f_grads.detach().cpu().numpy(), dtype=np.float32)
        f_nat_grad = lstsq(G, f_grads)[0]
        f_nat_grad = torch.from_numpy(np.array(f_nat_grad)).to(torch.float32).to('cuda')

        # one step of NGD
        self.grid_line_search_update(loss_function, f_nat_grad)
        self.param_groups[0]['params'] = self.params

        return loss