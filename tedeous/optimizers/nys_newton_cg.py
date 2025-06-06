"""The code is a copy from https://github.com/pratikrathore8/opt_for_pinns"""


import torch
from torch.optim import Optimizer
from torch.func import vmap
from functools import reduce
from torch.nn.utils import parameters_to_vector

def _armijo(f, x, gx, dx, t, alpha=0.1, beta=0.5):
    """Line search to find a step size that satisfies the Armijo condition."""
    f0 = f(x, 0, dx)
    f1 = f(x, t, dx)
    while f1 > f0 + alpha * t * gx.dot(dx):
        t *= beta
        f1 = f(x, t, dx)
    return t

def _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, x):
    """Applies the inverse of the Nystrom approximation of the Hessian to a vector."""
    z = U.T @ x
    z = (lambd_r + mu) * (U @ (S_mu_inv * z)) + (x - U @ z)
    return z

def _nystrom_pcg(hess, b, x, mu, U, S, r, tol, max_iters):
    """Solves a positive-definite linear system using NyströmPCG.

    `Frangella et al. Randomized Nyström Preconditioning. 
    SIAM Journal on Matrix Analysis and Applications, 2023.
    <https://epubs.siam.org/doi/10.1137/21M1466244>`"""
    lambd_r = S[r - 1]
    S_mu_inv = (S + mu) ** (-1)

    resid = b - (hess(x) + mu * x)
    with torch.no_grad():
        z = _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, resid)
        p = z.clone()

    i = 0

    while torch.norm(resid) > tol and i < max_iters:
        v = hess(p) + mu * p
        with torch.no_grad():
            alpha = torch.dot(resid, z) / torch.dot(p, v)
            x += alpha * p

            rTz = torch.dot(resid, z)
            resid -= alpha * v
            z = _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, resid)
            beta = torch.dot(resid, z) / rTz

            p = z + beta * p

        i += 1

    if torch.norm(resid) > tol:
        print(f"Warning: PCG did not converge to tolerance. Tolerance was {tol} but norm of residual is {torch.norm(resid)}")

    return x

class NysNewtonCG(Optimizer):
    """Implementation of NysNewtonCG, a damped Newton-CG method that uses Nyström preconditioning.
    
    `Rathore et al. Challenges in Training PINNs: A Loss Landscape Perspective.
    Preprint, 2024. <https://arxiv.org/abs/2402.01868>`

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    NOTE: This optimizer is currently a beta version. 

    Our implementation is inspired by the PyTorch implementation of `L-BFGS 
    <https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS>`.
    
    The parameters rank and mu will probably need to be tuned for your specific problem.
    If the optimizer is running very slowly, you can try one of the following:
    - Increase the rank (this should increase the accuracy of the Nyström approximation in PCG)
    - Reduce cg_tol (this will allow PCG to terminate with a less accurate solution)
    - Reduce cg_max_iters (this will allow PCG to terminate after fewer iterations)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1.0)
        rank (int, optional): rank of the Nyström approximation (default: 10)
        mu (float, optional): damping parameter (default: 1e-4)
        chunk_size (int, optional): number of Hessian-vector products to be computed in parallel (default: 1)
        cg_tol (float, optional): tolerance for PCG (default: 1e-16)
        cg_max_iters (int, optional): maximum number of PCG iterations (default: 1000)
        line_search_fn (str, optional): either 'armijo' or None (default: None)
        verbose (bool, optional): verbosity (default: False)
    
    """
    def __init__(self, params, lr=1.0, rank=10, mu=1e-4, chunk_size=1,
                 cg_tol=1e-16, cg_max_iters=1000, line_search_fn=None, verbose=False,precond_update_frequency=20,eigencdecomp_shift_attepmt_count=20):

        defaults = dict(lr=lr, rank=rank, chunk_size=chunk_size, mu=mu, cg_tol=cg_tol,
                        cg_max_iters=cg_max_iters, line_search_fn=line_search_fn, precond_update_frequency=precond_update_frequency,eigencdecomp_shift_attepmt_count=eigencdecomp_shift_attepmt_count)
        self.rank = rank
        self.mu = mu
        self.chunk_size = chunk_size
        self.cg_tol = cg_tol
        self.cg_max_iters = cg_max_iters
        self.line_search_fn = line_search_fn
        self.precond_update_frequency=precond_update_frequency
        self.eigencdecomp_shift_attepmt_count=eigencdecomp_shift_attepmt_count

        self.verbose = verbose
        self.U = None
        self.S = None
        self.n_iters = 0
        super(NysNewtonCG, self).__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError(
                "NysNewtonCG doesn't currently support per-parameter options (parameter groups)")

        if self.line_search_fn is not None and self.line_search_fn != 'armijo':
            raise ValueError("NysNewtonCG only supports Armijo line search")

        self._params = self.param_groups[0]['params']
        self._params_list = list(self._params)
        self._numel_cache = None
    

    def gradient(self, loss: torch.Tensor) -> torch.Tensor:
        """ Calculation of loss gradient by model parameters (NN, autograd)
            or model values (mat).

        Args:
            loss (torch.Tensor): result of loss calculation.

        Returns:
            torch.Tensor: calculated gradient vector.
        """
        dl_dparam = torch.autograd.grad(loss, self._params, create_graph=True)

        grads = parameters_to_vector(dl_dparam)

        return grads


    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns (i) the loss and (ii) gradient w.r.t. the parameters.
            The closure can compute the gradient w.r.t. the parameters by calling torch.autograd.grad on the loss with create_graph=True.
        """
        if self.n_iters == 0:
            # Store the previous direction for warm starting PCG
            self.old_dir = torch.zeros(
                self._numel(), device=self._params[0].device)

        # NOTE: The closure must return both the loss and the gradient
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss, grad_tuple = closure()

        g = torch.cat([grad.view(-1) for grad in grad_tuple if grad is not None])

        # One step update
        for group_idx, group in enumerate(self.param_groups):
            def hvp_temp(x):
                return self._hvp(g, self._params_list, x)

            # Calculate the Newton direction
            d = _nystrom_pcg(hvp_temp, g, self.old_dir,
                             self.mu, self.U, self.S, self.rank, self.cg_tol, self.cg_max_iters)

            # Store the previous direction for warm starting PCG
            self.old_dir = d

            # Check if d is a descent direction
            if torch.dot(d, g) <= 0:
                print("Warning: d is not a descent direction")

            if self.line_search_fn == 'armijo':
                x_init = self._clone_param()

                def obj_func(x, t, dx):
                    self._add_grad(t, dx)
                    loss = float(closure()[0])
                    self._set_param(x)
                    return loss

                # Use -d for convention
                t = _armijo(obj_func, x_init, g, -d, group['lr'])
            else:
                t = group['lr']

            self.state[group_idx]['t'] = t

            # update parameters
            ls = 0
            for p in group['params']:
                np = torch.numel(p)
                dp = d[ls:ls+np].view(p.shape)
                ls += np
                p.data.add_(-dp, alpha=t)

        self.n_iters += 1

        return loss, g

    def update_preconditioner(self, grad_tuple):
        """Update the Nystrom approximation of the Hessian.

        Args:
            grad_tuple (tuple): tuple of Tensors containing the gradients of the loss w.r.t. the parameters. 
            This tuple can be obtained by calling torch.autograd.grad on the loss with create_graph=True.
        """

        # Flatten and concatenate the gradients
        gradsH = torch.cat([gradient.view(-1)
                           for gradient in grad_tuple if gradient is not None])

        # Generate test matrix (NOTE: This is transposed test matrix)
        p = gradsH.shape[0]
        Phi = torch.randn(
            (self.rank, p), device=gradsH.device) / (p ** 0.5)
        Phi = torch.linalg.qr(Phi.t(), mode='reduced')[0].t()

        Y = self._hvp_vmap(gradsH, self._params_list)(Phi)

        # Calculate shift
        shift = torch.finfo(Y.dtype).eps
        Y_shifted = Y + shift * Phi

        # Calculate Phi^T * H * Phi (w/ shift) for Cholesky
        choleskytarget = torch.mm(Y_shifted, Phi.t())

        # Perform Cholesky, if fails, do eigendecomposition
        # The new shift is the abs of smallest eigenvalue (negative) plus the original shift
        try:
            C = torch.linalg.cholesky(choleskytarget)
        except:
            shift_attempt_count=0
            while shift_attempt_count<self.eigencdecomp_shift_attepmt_count:
                # eigendecomposition, eigenvalues and eigenvector matrix
                eigs, eigvectors = torch.linalg.eigh(choleskytarget)
                shift = shift + torch.abs(torch.min(eigs))
                # add shift to eigenvalues
                eigs = eigs + shift
                # put back the matrix for Cholesky by eigenvector * eigenvalues after shift * eigenvector^T
                try:
                    C = torch.linalg.cholesky(
                        torch.mm(eigvectors, torch.mm(torch.diag(eigs), eigvectors.T)))
                    break
                except:
                    shift_attempt_count+=1
                
        try:
            B = torch.linalg.solve_triangular(
                C, Y_shifted, upper=False, left=True)
        # temporary fix for issue @ https://github.com/pytorch/pytorch/issues/97211
        except:
            B = torch.linalg.solve_triangular(C.to('cpu'), Y_shifted.to(
                'cpu'), upper=False, left=True).to(C.device)
            
        # B = V * S * U^T b/c we have been using transposed sketch
        _, S, UT = torch.linalg.svd(B, full_matrices=False)
        self.U = UT.t()
        self.S = torch.max(torch.square(S) - shift, torch.tensor(0.0))

        self.rho = self.S[-1]

        if self.verbose:
            print(f'Approximate eigenvalues = {self.S}')

    def _hvp_vmap(self, grad_params, params):
        return vmap(lambda v: self._hvp(grad_params, params, v), in_dims=0, chunk_size=self.chunk_size)

    def _hvp(self, grad_params, params, v):
        Hv = torch.autograd.grad(grad_params, params, grad_outputs=v,
                                 retain_graph=True)
        Hv = tuple(Hvi.detach() for Hvi in Hv)
        return torch.cat([Hvi.reshape(-1) for Hvi in Hv])

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # Avoid in-place operation by creating a new tensor
            p.data = p.data.add(
                update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            # Replace the .data attribute of the tensor
            p.data = pdata.data