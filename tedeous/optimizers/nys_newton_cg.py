"""The code is a copy from https://github.com/pratikrathore8/opt_for_pinns"""

import torch
from torch.optim import Optimizer
from torch.func import vmap
from functools import reduce
from torch.nn.utils import parameters_to_vector


def randomized_svd(A, k, n_iter=2):
    """
    Performs Randomized Singular Value Decomposition (SVD) to reduce the dimensionality of the input matrix, which is useful as a preprocessing step when solving differential equations with neural networks. This method is more stable than full SVD, especially for ill-conditioned matrices, and helps to extract the most important features for training the neural network.
    
        Args:
            A (torch.Tensor): The input matrix to decompose.
            k (int): The number of singular values and vectors to return (rank of approximation).
            n_iter (int, optional): The number of power iterations to perform. Defaults to 2.
    
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - S (torch.Tensor): The top-k singular values.
                - UT (torch.Tensor): The top-k right singular vectors (transpose).
    
        Why:
        Randomized SVD helps reduce the complexity of the problem by extracting the most important features from the input data. This can improve the efficiency and accuracy of neural networks used to solve differential equations, especially when dealing with high-dimensional or noisy data.
    """
    m, n = A.shape
    k = min(k, min(m, n))  # Ensure k is valid
        
    # Random matrix for range finding
    Omega = torch.randn(n, k + 5, device=A.device, dtype=A.dtype)  # Slight oversampling
        
    # Power iteration for better approximation
    Y = A @ Omega
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
        
    # QR decomposition
    Q, _ = torch.linalg.qr(Y)
    Q = Q[:, :k+5]  # Take only what we need
        
    # Project matrix
    B_small = Q.T @ A
        
    # SVD of smaller matrix (much more stable)
    _, S_small, VT_small = torch.linalg.svd(B_small, full_matrices=False)
        
    # Reconstruct
    S = S_small[:k]
    UT = VT_small[:k, :]
        
    return S, UT


def _armijo(f, x, gx, dx, t, alpha=0.1, beta=0.5):
    """
    Finds an appropriate step size using the Armijo condition to ensure sufficient decrease in the loss function when updating the solution of a differential equation.
    
    This method refines the step size until the Armijo condition is met, ensuring that each iteration of the solver makes meaningful progress towards a solution.
    
    Args:
        f (callable): A function that evaluates the loss at a given point. It should accept the current state `x`, a step size `t`, and a direction `dx` as input.
        x (torch.Tensor): The current state of the solution.
        gx (torch.Tensor): The gradient of the loss function at the current state.
        dx (torch.Tensor): The search direction.
        t (float): The initial step size.
        alpha (float, optional): A parameter controlling the Armijo condition. Default is 0.1.
        beta (float, optional): A factor for reducing the step size. Default is 0.5.
    
    Returns:
        float: The adjusted step size that satisfies the Armijo condition.
    """
    f0 = f(x, 0, dx)
    f1 = f(x, t, dx)
    while f1 > f0 + alpha * t * gx.dot(dx):
        t *= beta
        f1 = f(x, t, dx)
    return t


def _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, x):
    """
    Applies the inverse of a neural-network-approximated Hessian to a given vector, which is a crucial step in optimizing the neural network's parameters to better fit the differential equation it's learning. This preconditioning step helps to accelerate the training process by improving the convergence rate.
    
        Args:
            U (torch.Tensor): The matrix of eigenvectors from the Nystrom approximation.
            S_mu_inv (torch.Tensor): The inverse of the eigenvalues matrix, adjusted by mu.
            mu (float): A regularization parameter.
            lambd_r (float): Another regularization parameter.
            x (torch.Tensor): The vector to which the inverse Hessian approximation is applied.
    
        Returns:
            torch.Tensor: The result of applying the inverse Nystrom approximation to the input vector.
    """
    z = U.T @ x
    z = (lambd_r + mu) * (U @ (S_mu_inv * z)) + (x - U @ z)
    return z


def _nystrom_pcg(hess, b, x, mu, U, S, r, tol, max_iters, verbose = False):
    """
    Solves a positive-definite linear system using a preconditioned conjugate gradient (PCG) method, enhanced with Nyström approximation for improved efficiency in the context of neural differential equation solvers. The method incorporates several numerical stability fixes to ensure reliable convergence when dealing with ill-conditioned problems or noisy gradients, which can arise during the training of neural networks to approximate solutions of differential equations.
    
    Args:
        hess (callable): A function that computes the Hessian-vector product.
        b (torch.Tensor): The right-hand side vector of the linear system.
        x (torch.Tensor): Initial guess for the solution vector.
        mu (float): Regularization parameter.
        U (torch.Tensor): Nyström basis vectors.
        S (torch.Tensor): Nyström eigenvalues.
        r (int): Rank of the Nyström approximation.
        tol (float): Tolerance for convergence.
        max_iters (int): Maximum number of iterations.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    
    Returns:
        torch.Tensor: The approximate solution to the linear system.
    
    WHY: This method is used to efficiently solve linear systems that arise during the training of neural networks for solving differential equations. The Nyström approximation reduces the computational cost, while the stability fixes ensure reliable convergence in the presence of numerical issues common in neural network training.
    """
    lambd_r = S[r - 1]
    S_mu_inv = (S + mu) ** (-1)
    
    # Fix 1: Add numerical stability checks
    if torch.isnan(x).any(): 
        if verbose: print('x is nan - resetting to zero')
        x = torch.zeros_like(x)
    if torch.isnan(b).any(): 
        if verbose: print('b is nan')
        return x
    
    # Fix 2: Check condition number and adjust mu if needed
    condition_estimate = S[0] / (S[-1] + mu) if len(S) > 0 else 1.0
    if condition_estimate > 1e12:
        mu_adjusted = mu * 10
        if verbose: print(f'Poor conditioning detected (cond~{condition_estimate:.2e}), increasing mu from {mu} to {mu_adjusted}')
        S_mu_inv = (S + mu_adjusted) ** (-1)
        mu = mu_adjusted
    
    hess_x = hess(x)
    if torch.isnan(hess_x).any(): 
        if verbose: print('hess(x) is nan')
        return x
        
    resid = b - (hess_x + mu * x)
    if torch.isnan(resid).any(): 
        if verbose: print('resid is nan')
        return x
    
    # Fix 3: Check initial residual norm
    initial_resid_norm = torch.norm(resid)
    if initial_resid_norm > 1e6:
        if verbose: print(f'Warning: Large initial residual norm {initial_resid_norm:.2e}')
        # Scale down the problem
        scale_factor = 1e6 / initial_resid_norm
        b = b * scale_factor
        resid = resid * scale_factor
        if verbose: print(f'Scaling problem by {scale_factor:.2e}')

    with torch.no_grad():
        z = _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, resid)
        p = z.clone()
    
    i = 0
    prev_resid_norm = torch.norm(resid)
    stagnation_count = 0
    
    while torch.norm(resid) > tol and i < max_iters:
        v = hess(p) + mu * p
        
        # Fix 4: Check for NaN in critical vectors
        if torch.isnan(p).any():
            if verbose: print(f'p is NaN at iteration {i}')
            break
        if torch.isnan(v).any():
            if verbose: print(f'v is NaN at iteration {i}')
            break
            
        with torch.no_grad():
            # Fix 5: Check denominator before division
            pv_dot = torch.dot(p, v)
            rz_dot = torch.dot(resid, z)
            
            if torch.isnan(pv_dot) or torch.abs(pv_dot) < 1e-16:
                if verbose: print(f'Bad denominator: pv_dot = {pv_dot}, breaking PCG')
                break
            if torch.isnan(rz_dot):
                if verbose: print(f'Bad numerator: rz_dot = {rz_dot}, breaking PCG')
                break
                
            alpha = rz_dot / pv_dot
            
            # Fix 6: Clamp alpha to reasonable range
            if torch.abs(alpha) > 1e6:
                if verbose: print(f'Large alpha detected: {alpha:.2e}, clamping')
                alpha = torch.sign(alpha) * 1e6
            
            x_new = x + alpha * p
            
            # Fix 7: Check for NaN in updated x
            if torch.isnan(x_new).any():
                if verbose: print(f'x becomes NaN at iteration {i}, alpha = {alpha:.2e}')
                if verbose: print(f'rz_dot = {rz_dot:.2e}, pv_dot = {pv_dot:.2e}')
                break
            
            x = x_new
            rTz = rz_dot
            resid_new = resid - alpha * v
            
            # Fix 8: Check for residual growth/stagnation
            current_resid_norm = torch.norm(resid_new)
            if current_resid_norm > 2 * prev_resid_norm:
                #print(f'Residual growing: {prev_resid_norm:.2e} -> {current_resid_norm:.2e}')
                stagnation_count += 1
                if stagnation_count > 3:
                    if verbose: print('Too many residual growth steps, breaking')
                    break
            elif abs(current_resid_norm - prev_resid_norm) / prev_resid_norm < 1e-10:
                stagnation_count += 1
                if stagnation_count > 5:
                    if verbose: print('PCG stagnating, breaking')
                    break
            else:
                stagnation_count = 0
                
            resid = resid_new
            prev_resid_norm = current_resid_norm
            
            z = _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, resid)
            
            # Fix 9: Check z for NaN
            if torch.isnan(z).any():
                if verbose: print(f'z becomes NaN at iteration {i}')
                break
                
            beta = torch.dot(resid, z) / rTz
            
            # Fix 10: Clamp beta
            if torch.abs(beta) > 1e6:
                if verbose: print(f'Large beta detected: {beta:.2e}, clamping')
                beta = torch.sign(beta) * 1e6
                
            p = z + beta * p
            
        i += 1
    
    final_resid_norm = torch.norm(resid)
    if final_resid_norm > tol:
        print(f"Warning: PCG did not converge. Tolerance: {tol}, Final residual: {final_resid_norm:.2e}, Iterations: {i}")
    
    return x


class NysNewtonCG(Optimizer):
    """
    Implements a damped Newton-CG method with Nyström preconditioning for optimization tasks. This method combines the global convergence of Newton's method with the efficiency of conjugate gradient descent, enhanced by Nyström approximation for preconditioning.
    
    
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
                 cg_tol=1e-16, cg_max_iters=1000, line_search_fn=None,
                 verbose=False, precond_update_frequency=20,
                 eigencdecomp_shift_attepmt_count=20):
        """
        Initialize the NysNewtonCG optimizer.
        
                This optimizer approximates the inverse Hessian using the Nyström method and solves the Newton step with a conjugate gradient solver.
                It is designed to efficiently solve differential equations by optimizing the parameters of a neural network.
        
                Args:
                    params: Iterable of parameters to optimize or dicts defining parameter groups.
                    lr: Learning rate (default: 1.0).
                    rank: Rank of the Nyström approximation (default: 10).
                    mu: Regularization parameter for the Hessian approximation (default: 1e-4).
                    chunk_size: Chunk size for processing parameters (default: 1).
                    cg_tol: Tolerance for the conjugate gradient solver (default: 1e-16).
                    cg_max_iters: Maximum number of iterations for the conjugate gradient solver (default: 1000).
                    line_search_fn: Line search function to use (default: None, only 'armijo' is supported).
                    verbose: Whether to print debugging information (default: False).
                    precond_update_frequency: How often to update the preconditioner (default: 20).
                    eigencdecomp_shift_attepmt_count: Number of attempts to shift the eigenvalue decomposition (default: 20).
        
                Returns:
                    None
        
                Class Fields:
                    rank (int): Rank of the Nyström approximation.
                    mu (float): Regularization parameter for the Hessian approximation.
                    chunk_size (int): Chunk size for processing parameters.
                    cg_tol (float): Tolerance for the conjugate gradient solver.
                    cg_max_iters (int): Maximum number of iterations for the conjugate gradient solver.
                    line_search_fn (str or None): Line search function to use.
                    precond_update_frequency (int): How often to update the preconditioner.
                    eigencdecomp_shift_attepmt_count (int): Number of attempts to shift the eigenvalue decomposition.
                    verbose (bool): Whether to print debugging information.
                    U (torch.Tensor or None): The U matrix from the SVD of the Nyström approximation. Initialized to None.
                    S (torch.Tensor or None): The singular values from the SVD of the Nyström approximation. Initialized to None.
                    n_iters (int): The number of iterations performed. Initialized to 0.
                    _params (list): List of parameters to optimize.
                    _params_list (list): List of parameters to optimize.
                    _numel_cache (int): Cached number of elements in the parameters. Initialized to None.
        """

        defaults = dict(lr=lr, rank=rank, chunk_size=chunk_size, mu=mu, cg_tol=cg_tol,
                        cg_max_iters=cg_max_iters, line_search_fn=line_search_fn,
                        precond_update_frequency=precond_update_frequency,
                        eigencdecomp_shift_attepmt_count=eigencdecomp_shift_attepmt_count)
        self.rank = rank
        self.mu = mu
        self.chunk_size = chunk_size
        self.cg_tol = cg_tol
        self.cg_max_iters = cg_max_iters
        self.line_search_fn = line_search_fn
        self.precond_update_frequency = precond_update_frequency
        self.eigencdecomp_shift_attepmt_count = eigencdecomp_shift_attepmt_count

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
        """
        Calculates the gradient of the loss with respect to the model's parameters.
        
        This gradient is essential for optimizing the neural network to better approximate the solution of the differential equation.
        It leverages PyTorch's automatic differentiation capabilities to efficiently compute the gradient of the loss function with respect to the model parameters.
        
        Args:
            loss (torch.Tensor): The calculated loss value, representing the error between the neural network's prediction and the true solution (or a proxy thereof) of the differential equation.
        
        Returns:
            torch.Tensor: A vector containing the gradients of the loss with respect to each of the model's parameters. This vector is used to update the model's parameters during the optimization process, iteratively improving the solution.
        """
        # torch.autograd.set_detect_anomaly(True)
        dl_dparam = torch.autograd.grad(loss, self._params, create_graph=True)

        grads = parameters_to_vector(dl_dparam)

        return grads

    def step(self, closure=None):
        """
        Perform a single optimization step to refine the neural network's approximation of the differential equation solution. This involves updating the network's parameters based on the computed gradients and a preconditioner to accelerate convergence. The method incorporates a line search to ensure stable and effective updates, ultimately minimizing the discrepancy between the network's output and the true solution of the differential equation.
        
                Args:
                    closure (callable, optional): A closure that reevaluates the model and returns (i) the loss and (ii) gradient w.r.t. the parameters.
                    The closure can compute the gradient w.r.t. the parameters by calling torch.autograd.grad on the loss with create_graph=True.
        
                Returns:
                    tuple: A tuple containing the loss and the concatenated gradients.
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

        if self.n_iters % self.precond_update_frequency == 0:
            if self.verbose: print('here t={} and freq={}'.format(self.n_iters, self.precond_update_frequency))
            self.update_preconditioner(grad_tuple)

        g = torch.cat([grad.view(-1) for grad in grad_tuple if grad is not None])

        # One step update
        for group_idx, group in enumerate(self.param_groups):
            def hvp_temp(x):
                return self._hvp(g, self._params_list, x)

            # Calculate the Newton direction
            d = _nystrom_pcg(hvp_temp, g, self.old_dir,
                             self.mu, self.U, self.S, self.rank, self.cg_tol, self.cg_max_iters, verbose = self.verbose)

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
                dp = d[ls:ls + np].view(p.shape)
                ls += np
                p.data.add_(-dp, alpha=t)

        self.n_iters += 1

        return loss, g

    def update_preconditioner(self, grad_tuple):
        """
        Update the low-rank Nyström approximation of the Hessian based on the gradient information. This approximation is crucial for preconditioning optimization, effectively capturing the curvature of the loss landscape to accelerate convergence.
        
                Args:
                    grad_tuple (tuple): Tuple of Tensors containing the gradients of the loss with respect to the parameters.
                        This tuple is obtained by calling `torch.autograd.grad` on the loss with `create_graph=True`.
        
                Returns:
                    None: The method updates the internal state of the preconditioner (U, S, rho) to reflect the new approximation.
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
            shift_attempt_count = 0
            while shift_attempt_count < self.eigencdecomp_shift_attepmt_count:
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
                    shift_attempt_count += 1

        try:
            B = torch.linalg.solve_triangular(
                C, Y_shifted, upper=False, left=True)
        # temporary fix for issue @ https://github.com/pytorch/pytorch/issues/97211
        except:
            B = torch.linalg.solve_triangular(C.to('cpu'), Y_shifted.to(
                'cpu'), upper=False, left=True).to(C.device)

        # B = V * S * U^T b/c we have been using transposed sketch
        #_, S, UT = torch.linalg.svd(B, full_matrices=False)
        S, UT = randomized_svd(B, self.rank,n_iter = 2)

        self.U = UT.t()
        self.S = torch.max(torch.square(S) - shift, torch.tensor(0.0))

        self.rho = self.S[-1]

        if self.verbose:
            print(f'Approximate eigenvalues = {self.S}')

    def _hvp_vmap(self, grad_params, params):
        """
        Computes a vmap of the Hessian-vector product. This is done to efficiently compute multiple Hessian-vector products in parallel, which is useful for training neural networks to solve differential equations. By vectorizing the computation, we can leverage the parallel processing capabilities of GPUs or TPUs, significantly speeding up the training process.
        
                Args:
                    grad_params: The gradient of the loss function with respect to the parameters.
                    params: The parameters of the model.
        
                Returns:
                    The vmap of the Hessian-vector product.
        """
        return vmap(lambda v: self._hvp(grad_params, params, v), in_dims=0, chunk_size=self.chunk_size)

    def _hvp(self, grad_params, params, v):
        """
        Compute the Hessian-vector product to approximate the curvature of the loss landscape.
        
                This method calculates the product of the Hessian of the loss function
                with respect to the model parameters and a given vector. This is a key step
                in applying Newton-based optimization methods, as it allows us to efficiently
                estimate the change in the gradient direction.
        
                Args:
                  grad_params: Gradients of the loss function with respect to the parameters.
                  params: Model parameters.
                  v: Vector to multiply with the Hessian.
        
                Returns:
                  torch.Tensor: The Hessian-vector product, flattened into a 1D tensor.
        """
        Hv = torch.autograd.grad(grad_params, params, grad_outputs=v,
                                 retain_graph=True)
        Hv = tuple(Hvi.detach() for Hvi in Hv)
        return torch.cat([Hvi.reshape(-1) for Hvi in Hv])

    def _numel(self):
        """
        Calculates and returns the total number of elements in the model's parameters.
        
                This method computes the sum of the number of elements in each
                parameter stored in the object. It uses a cache to store the result
                for subsequent calls to optimize performance during the training process
                of neural differential equation solvers.
        
                Args:
                    self: The object instance.
        
                Returns:
                    int: The total number of elements in all parameters.
        """
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _add_grad(self, step_size, update):
        """
        Adds a scaled gradient update to the model's parameters.
        
                This method iterates through the model's parameters, applying the provided update scaled by the given step size. This is a crucial step in adjusting the network's weights based on the calculated gradients during the optimization process, bringing the model closer to the solution of the differential equation.
        
                Args:
                  step_size: The scaling factor for the update, controlling the magnitude of the parameter adjustment.
                  update: A 1D tensor containing the gradient update for all parameters, representing the direction and magnitude of change.
        
                Returns:
                  None. This method modifies the parameters in place.
        """
        offset = 0
        for p in self._params:
            numel = p.numel()
            # Avoid in-place operation by creating a new tensor
            p.data = p.data.add(
                update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        """
        Clones the parameters of the module for neural differential equation solving.
        
                This method creates a new list containing clones of each parameter
                in the module's internal parameter list `_params`. The clones are
                created with a contiguous memory format to ensure optimal performance
                during the neural network's training process for solving differential equations.
                This is crucial for maintaining a consistent parameter set when evaluating the loss function
                and updating the network's weights.
        
                Args:
                    self: The module instance.
        
                Returns:
                    list: A list containing clones of the module's parameters,
                    each with a contiguous memory format.
        """
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        """
        Sets the module's parameters using provided data.
        
                This method updates the module's parameters with new values, ensuring the model adapts to the evolving solution landscape during the training process. This is crucial for accurately approximating the differential equation's solution.
        
                Args:
                    params_data: An iterable containing the updated parameter data, typically obtained during an optimization step.
        
                Returns:
                    None. The method modifies the module's parameters in place.
        """
        for p, pdata in zip(self._params, params_data):
            # Replace the .data attribute of the tensor
            p.data = pdata.data
