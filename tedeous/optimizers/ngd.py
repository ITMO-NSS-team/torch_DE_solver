import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tedeous.utils import replace_none_by_zero
from tedeous.device import check_device


class NGD(torch.optim.Optimizer):
    """
    NGD implementation (https://arxiv.org/abs/2302.13163).
    """

    """NGD implementation (https://arxiv.org/abs/2302.13163).
    """

    def __init__(self, params,
                 grid_steps_number: int = 30):
        """
        Initializes the Natural Gradient Descent optimizer.
        
                This optimizer is designed to refine the optimization process when training neural networks to solve differential equations. It sets up the optimization landscape by defining a grid of steps, which influences how the model parameters are updated during training. This approach helps in navigating the complex loss surfaces often encountered when solving differential equations with neural networks.
        
                Args:
                    params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
                    grid_steps_number (int, optional):  Determines the granularity of the search space for optimization. A higher number allows for finer adjustments to the parameters. Defaults to 30.
        
                Returns:
                    None
        """
        defaults = {'grid_steps_number': grid_steps_number}
        super(NGD, self).__init__(params, defaults)
        self.params = self.param_groups[0]['params']
        self.grid_steps_number = grid_steps_number
        self.grid_steps = torch.linspace(0, self.grid_steps_number, self.grid_steps_number + 1)
        self.steps = 0.5**self.grid_steps
        self.cuda_out_of_memory_flag=False
        self.cuda_empty_once_for_test=True

    def grid_line_search_update(self, loss_function: callable, f_nat_grad: torch.Tensor) -> None:
        """
        Update model parameters using a grid line search along the natural gradient direction to minimize the loss function. This method explores different step sizes to find the optimal update that best approximates the solution to the differential equation.
        
                Args:
                    loss_function (callable): A callable that computes the loss value. It should return a tuple of (loss, aux_variables).
                    f_nat_grad (torch.Tensor): The natural gradient to update the parameters with.
        
                Returns:
                    None. The model parameters are updated in place.
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
        """
        Computes the Gram matrix of the Jacobian of the PDE residuals with respect to the model parameters.
        
        This matrix is a key component in various optimization strategies for training neural networks to solve differential equations.
        It provides information about the sensitivity of the residuals to changes in the parameters,
        which is then used to improve convergence and stability during training.
        
        Args:
            residuals (torch.Tensor): The PDE residual values evaluated at different points.
        
        Returns:
            torch.Tensor: The Gram matrix, a measure of the correlation between the gradients of the residuals with respect to the parameters.
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


    def gram_factory_cpu(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Computes the Gram matrix of the PDE residuals' Jacobian with respect to the model parameters.
        
        This matrix is used to analyze the sensitivity of the residuals to changes in the network's parameters.
        It provides insights into the optimization landscape and can be used to improve the training process.
        
        Args:
            residuals (torch.Tensor): The PDE residual values evaluated at different points.
        
        Returns:
            torch.Tensor: The Gram matrix, a measure of the correlation between the gradients of the residuals.
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

        J = jacobian().cpu()
        return 1.0 / len(residuals) * J.T @ J


    
    def torch_cuda_lstsq(self, A: torch.Tensor, B: torch.Tensor, tol: float = None) -> torch.Tensor:
        """
        Find the least-squares solution for a system of linear equations represented by torch.Tensor on a CUDA device. This method is used to optimize the neural network's approximation of the differential equation's solution by minimizing the residual error.
        
                Args:
                    A (torch.Tensor): The left-hand side tensor of shape (*, m, n), where * represents zero or more batch dimensions. Represents the coefficients in the linear system.
                    B (torch.Tensor): The right-hand side tensor of shape (*, m, k), where * represents zero or more batch dimensions. Represents the constants in the linear system.
                    tol (float, optional): Tolerance value used to determine the effective rank of A. Defaults to the machine precision of the dtype of A if not provided.
        
                Returns:
                    torch.Tensor: The least-squares solution for A and B, obtained via Singular Value Decomposition (SVD). This solution minimizes the error in satisfying the linear system, contributing to a more accurate neural network approximation of the differential equation's solution.
        """
        tol = torch.finfo(A.dtype).eps if tol is None else tol
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        Spinv = torch.zeros_like(S)
        Spinv[S>tol] = 1/S[S>tol]
        UhB = U.adjoint() @ B
        if Spinv.ndim!=UhB.ndim:
            Spinv = Spinv.unsqueeze(-1)
        SpinvUhB = Spinv * UhB
        return Vh.adjoint() @ SpinvUhB



    def numpy_lstsq(self, A: torch.Tensor, B: torch.Tensor, rcond: float = None) -> torch.Tensor:
        """
        Computes the least squares solution to a linear matrix equation using NumPy to approximate solutions of differential equations.
        
                This method takes two PyTorch tensors, converts them to NumPy arrays,
                uses NumPy's `linalg.lstsq` to solve the least squares problem, and
                then converts the result back to a PyTorch tensor. The resulting tensor
                is then placed on the correct device. This is a crucial step in ensuring
                compatibility with the broader neural differential equation solving framework,
                allowing for seamless integration of the computed solution within the
                PyTorch-based training and evaluation pipelines.
        
                Args:
                    A: The "coefficient" matrix (left-hand side of the equation) as a PyTorch tensor.
                    B: The "dependent variable" values (right-hand side of the equation) as a PyTorch tensor.
                    rcond:  Cutoff ratio for small singular values of a.
                        For the purposes of rank determination, singular values are treated
                        as zero if they are smaller than rcond times the largest singular
                        value of a.
        
                Returns:
                    torch.Tensor: The least squares solution, converted back to a PyTorch tensor
                    and placed on the correct device using the `check_device` function.
        """

        A = A.detach().cpu().numpy()
        B = B.detach().cpu().numpy()

        f_nat_grad = np.linalg.lstsq(A, B,rcond=rcond)[0] 

        f_nat_grad=torch.from_numpy(f_nat_grad)

        f_nat_grad = check_device(f_nat_grad)

        return f_nat_grad


    def step(self, closure=None) -> torch.Tensor:
        """
        Runs one step of the Natural Gradient Descent (NGD) optimization.
        
                This method computes the natural gradient and updates the model parameters
                to minimize the loss function, effectively solving the differential equation
                by optimizing the neural network's parameters.
        
                Args:
                    closure (callable, optional): A closure that reevaluates the model and
                        returns the loss. It should return a tuple containing intermediate
                        results, boundary values, true boundary values, the loss tensor,
                        and the loss function itself.
        
                Returns:
                    torch.Tensor: The loss value after the NGD step.
        """

        int_res, bval, true_bval, loss, loss_function = closure()
        grads = torch.autograd.grad(loss, self.params, retain_graph=True, allow_unused=True)
        grads = replace_none_by_zero(grads)
        f_grads = parameters_to_vector(grads)

        bound_res = bval-true_bval

        ## assemble gramian
        #G_int  = self.gram_factory(int_res.reshape(-1))
        #G_bdry = self.gram_factory(bound_res.reshape(-1))
        #G      = G_int + G_bdry

        ## Marquardt-Levenberg
        #Id = torch.eye(len(G))
        #G = torch.min(torch.tensor([loss, 0.0])) * Id + G

        

        # compute natural gradient
        if not self.cuda_out_of_memory_flag:
            try:
                if self.cuda_empty_once_for_test:
                    #print('Initial GPU check')
                    torch.cuda.empty_cache()
                    self.cuda_empty_once_for_test=False
                
                # assemble gramian

                #print('NGD GPU step')

                G_int  = self.gram_factory(int_res.reshape(-1))
                G_bdry = self.gram_factory(bound_res.reshape(-1))
                G      = G_int + G_bdry

                # Marquardt-Levenberg
                Id = torch.eye(len(G))
                G = torch.min(torch.tensor([loss, 0.0])) * Id + G

                f_nat_grad = self.torch_cuda_lstsq(G, f_grads)   
            except torch.OutOfMemoryError:
                print('[Warning] Least square returned CUDA out of memory error, CPU and RAM are used, which is significantly slower')
                self.cuda_out_of_memory_flag=True

                G_int  = self.gram_factory_cpu(int_res.reshape(-1).cpu())
                G_bdry = self.gram_factory_cpu(bound_res.reshape(-1).cpu())
                G      = G_int + G_bdry


                f_nat_grad = self.numpy_lstsq(G, f_grads)
        else:


            #print('NGD CPU step')

            G_int  = self.gram_factory_cpu(int_res.reshape(-1).cpu())
            G_bdry = self.gram_factory_cpu(bound_res.reshape(-1).cpu())
            G      = G_int + G_bdry

            f_nat_grad = self.numpy_lstsq(G, f_grads)

        # one step of NGD
        self.grid_line_search_update(loss_function, f_nat_grad)
        self.param_groups[0]['params'] = self.params

        return loss