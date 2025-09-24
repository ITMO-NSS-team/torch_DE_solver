from typing import Tuple
import torch
from copy import copy
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tedeous.device import device_type


class CSO(torch.optim.Optimizer):
    """
    Custom CSO optimizer.
    """

    """Custom CSO optimizer.
    """

    def __init__(self,
                 params,
                 pop_size: int = 20,
                 fi: float = 0.0,
                 lr: float = 1e-3,
                 betas: Tuple = (0.99, 0.999),
                 c_decrease: bool = False,
                 variance: float = 1,
                 epsilon: float = 1e-8,
                 n_iter: int = 2000):
        """
        Initializes the Competitive Swarm Optimizer (CSO) for training neural networks to solve differential equations.
        
        The CSO leverages a population of particles (swarm) to explore the parameter space of a neural network,
        guiding it towards a solution that satisfies the differential equation. This initialization sets up the swarm
        with specified parameters, preparing it for the iterative optimization process. The parameters control aspects
        such as swarm size, influence of particle positions, and learning rates, allowing for fine-tuning of the
        optimization process to effectively solve the target differential equation.
        
        Args:
            params (torch.nn.Parameter): Model parameters to be optimized.
            pop_size (int, optional): Population of the CSO swarm. Defaults to 20.
            fi (float, optional): Parameter that controls the influence of mean position value of the
                relevant particles. Defaults to 0 for pop_size < 100.
            lr (float, optional): Learning rate for gradient descent. Defaults to 1e-3.
            betas (tuple(float, float), optional): Coefficients used for computing
                running averages of gradient and its square. Defaults to (0.99, 0.999).
            c_decrease (bool, optional): Flag for update_pso_params method. Defaults to False.
            variance (float, optional): Variance parameter for swarm creation
                based on model. Defaults to 1.
            epsilon (float, optional): Term added to improve the numerical stability.
                Defaults to 1e-8.
            n_iter (int, optional): Number of iterations. Defaults to 2000.
        """
        defaults = {'pop_size': pop_size,
                    'fi': fi, 
                    'lr': lr, 'betas': betas,
                    'c_decrease': c_decrease,
                    'variance': variance,
                    'epsilon': epsilon}
        super(CSO, self).__init__(params, defaults)
        self.params = self.param_groups[0]['params']
        self.pop_size = pop_size
        self.fi = fi
        self.epsilon = epsilon
        self.beta1, self.beta2 = betas
        self.lr = lr * np.sqrt(1 - self.beta2) / (1 - self.beta1)
        self.use_grad = True if self.lr != 0 else False
        self.variance = variance
        self.name = "СSO"
        self.n_iter = n_iter
        self.t = 0
        self.V_max = 1.0

        vec_shape = self.params_to_vec().shape
        self.vec_shape = list(vec_shape)[0]

        self.swarm = self.build_swarm()

        self.p = copy(self.swarm).detach()

        self.v = self.start_velocities()
        self.m1 = torch.zeros(self.pop_size, self.vec_shape)
        self.m2 = torch.zeros(self.pop_size, self.vec_shape)

        self.indicator = True

    def params_to_vec(self) -> torch.Tensor:
        """
        Converts the model's parameters or values into a single vector.
        
        This is useful for optimization and analysis within the neural differential equation solving process,
        allowing the model's state to be represented in a compact form suitable for gradient-based methods.
        
        Args:
            None
        
        Returns:
            torch.Tensor: A 1D tensor containing all model parameters or values.
        """
        if not isinstance(self.params, torch.Tensor):
            vec = parameters_to_vector(self.params)
        else:
            self.model_shape = self.params.shape
            vec = self.params.reshape(-1)

        return vec

    def vec_to_params(self, vec: torch.Tensor) -> None:
        """
        Converts a vector into the neural network's parameters.
        
        This method updates the model's parameters using the provided vector, 
        allowing the optimization process to explore different parameter configurations 
        for solving the differential equation. It ensures that the vector is properly 
        reshaped and applied to the model's parameters, whether they are structured 
        as a collection of PyTorch parameters or a single tensor.
        
        Args:
            vec (torch.Tensor): A vector representing a potential set of parameters for the neural network.
        
        Returns:
            None
        """
        if not isinstance(self.params, torch.Tensor):
            vector_to_parameters(vec, self.params)
        else:
            self.params.data = vec.reshape(self.params).data


    def build_swarm(self):
        """
        Initializes a population of potential solutions (swarm) for training a neural network to solve a differential equation.
        
        The swarm is created by perturbing a base solution vector with random noise. This creates a diverse set of initial guesses for the network's parameters,
        allowing the optimization process to explore the solution space more effectively. The first particle in the swarm is set to the original solution vector.
        
        Args:
            None
        
        Returns:
            torch.Tensor: The initialized swarm. Each row represents a particle (a set of neural network parameters),
                          with gradients enabled for optimization.
        """
        vector = self.params_to_vec()
        matrix = []
        for _ in range(self.pop_size):
            matrix.append(vector.reshape(1, -1))
        matrix = torch.cat(matrix)
        variance = torch.FloatTensor(self.pop_size, self.vec_shape).uniform_(
            -self.variance, self.variance).to(device_type())
        swarm = matrix + variance
        swarm[0] = matrix[0]
        return swarm.clone().detach().requires_grad_(True)

    def start_velocities(self) -> torch.Tensor:
        """
        Initializes particle velocities to zero, ensuring no initial bias in the solution space when solving differential equations using neural networks.
        
        Args:
            None
        
        Returns:
            torch.Tensor: A tensor of zeros representing the initial velocities of the particles, shaped according to the population size and vector shape.
        """
        return torch.zeros((self.pop_size, self.vec_shape))

    def gradient(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Calculates the gradient of the loss with respect to the model parameters.
        
        This gradient is crucial for updating the model's parameters during the training process,
        allowing the neural network to learn and approximate the solution to the differential equation.
        The gradient indicates the direction and magnitude of the change needed in the parameters to minimize the loss function.
        
        Args:
            loss (torch.Tensor): The calculated loss value, representing the error between the model's prediction and the true solution.
        
        Returns:
            torch.Tensor: A vector containing the gradients of the loss with respect to each model parameter.
        """
        dl_dparam = torch.autograd.grad(loss, self.params)

        grads = parameters_to_vector(dl_dparam)

        return grads
    
    def update_zero_v(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates velocity values that are zero to prevent stagnation in the optimization process.
        
                Zero velocities can hinder the exploration of the solution space when training neural networks to solve differential equations. This method replaces these zero values with random values scaled by the variance and `V_max`, with a randomized sign, to encourage continued exploration and avoid getting stuck in local minima.
        
                Args:
                    self: The instance of the CSO class.
        
                Returns:
                    Tuple[torch.Tensor, torch.Tensor]: A tuple containing the updated velocity tensor (`self.v`) and the variance tensor (`self.variance`).
        """

        idx = torch.where(abs(self.v) == 0)
        if len(idx[0]) > 0: 
            rand = torch.rand_like(self.v[idx])  # Генерируем случайные значения той же формы, что и self.v
            self.v[idx] = torch.where(rand > 0.5, rand * self.variance * (-self.V_max), rand * self.variance * self.V_max)

    def get_randoms(self) -> torch.Tensor:
        """
        Generate random values to update the particles' positions. These random values are crucial for exploring the solution space during the optimization process, allowing the particles to move and potentially discover better solutions for the differential equation.
        
        Args:
            None
        
        Returns:
            torch.Tensor: A tensor of random values with shape (2, 1, self.vec_shape).
        """
        return torch.rand((2, 1, self.vec_shape))

    def update_p_best(self) -> None:
        """
        Updates the personal best positions of particles within the swarm.
        
                This method refines the individual search trajectories of each particle
                by comparing their current positions with their best historical positions.
                If a particle has found a better position (lower loss) than its previous
                best, its personal best position is updated to reflect this improvement.
        
                Args:
                    None
        
                Returns:
                    None
        """

        idx = torch.where(self.loss_swarm < self.f_p)

        self.p[idx] = self.swarm[idx]
        self.f_p[idx] = self.loss_swarm[idx].detach()

    def update_g_best(self) -> None:
        """
        Updates the globally best position (*g-best*) among all particles. This ensures that the optimization process converges towards the best solution found across the entire swarm, which is crucial for effectively training neural networks to approximate solutions to differential equations.
        
            Args:
                None
        
            Returns:
                None
        """
        self.g_best = self.p[torch.argmin(self.f_p)]

    def gradient_descent(self) -> torch.Tensor:
        """
        Updates the gradient based on the Adam optimization algorithm.
        
        This method refines the gradient using a combination of momentum and adaptive learning rates,
        enhancing the training process of the neural network used to approximate the solution of
        the differential equation. It calculates the update term based on the exponentially
        weighted moving averages of the gradients and their squares. This helps to improve the
        convergence and stability of the training process.
        
        Args:
            None
        
        Returns:
            torch.Tensor: The calculated update term to be applied to the network's parameters.
        """
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * self.grads_swarm
        self.m2 = self.beta2 * self.m2 + (1 - self.beta2) * torch.square(
            self.grads_swarm)

        update = self.lr * self.m1 / (torch.sqrt(torch.abs(self.m2)) + self.epsilon)

        return update

    def step(self, closure=None) -> torch.Tensor:
        """
        Runs one optimization step to refine the neural network's approximation of the differential equation solution. It adjusts particle positions based on a competitive swarm approach, leveraging loss gradients and velocity updates to explore the solution space.
        
                Args:
                    closure (callable, optional): A function that evaluates the loss and gradients of the swarm. Defaults to None.
        
                Returns:
                    torch.Tensor: The minimum loss value achieved by the swarm, representing the best approximation of the differential equation's solution found so far. This value indicates the accuracy of the neural network's solution at the current optimization step.
        """
        self.loss_swarm, self.grads_swarm = closure()
        if self.indicator:
            self.f_p = copy(self.loss_swarm).detach()
            self.g_best = self.p[torch.argmin(self.f_p)]
            self.indicator = False

        fix_attempt=0
        while torch.any(self.loss_swarm!=self.loss_swarm):
            self.swarm=self.swarm+0.001*torch.rand(size=self.swarm.shape)
            self.loss_swarm, self.grads_swarm = closure()
            fix_attempt+=1
            if fix_attempt>5:
                break

        U = list(range(self.pop_size))
        while U:
            i, j = np.random.choice(U, 2, replace=False)
            X1, X2 = self.swarm[i], self.swarm[j]
            
            if self.loss_swarm[i] <= self.loss_swarm[j]:
                Xw, Xl = X1, X2
                iw, il = i, j
            else:
                Xw, Xl = X2, X1
                iw, il = j, i

            R1, R2, R3 = torch.rand(self.vec_shape), torch.rand(self.vec_shape), torch.rand(self.vec_shape)
            self.v[il] = R1 * self.v[il] + R2 * (Xw - Xl) + self.fi * R3 * ((Xl + Xw)/2 - Xl)
                
            with torch.no_grad():
                self.swarm[il] = Xl + self.v[il]
            U.remove(i)
            U.remove(j)

        if self.use_grad: 
            self.swarm = self.swarm -  self.gradient_descent()

        self.update_p_best()
        self.update_g_best()

        if self.t > 150:
            self.update_zero_v()
        self.vec_to_params(self.g_best)
        min_loss =  torch.min(self.f_p)
        self.t += 1
        return min_loss
