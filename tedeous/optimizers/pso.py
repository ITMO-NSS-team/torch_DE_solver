from typing import Tuple
import torch
from copy import copy
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tedeous.device import device_type


class PSO(torch.optim.Optimizer):
    """
    Custom PSO optimizer.
    """

    """Custom PSO optimizer.
    """

    def __init__(self,
                 params,
                 pop_size: int = 30,
                 b: float = 0.9,
                 c1: float = 8e-2,
                 c2: float = 5e-1,
                 lr: float = 1e-3,
                 betas: Tuple = (0.99, 0.999),
                 c_decrease: bool = False,
                 variance: float = 1,
                 epsilon: float = 1e-8,
                 n_iter: int = 2000):
        """
        Initializes the Particle Swarm Optimizer (PSO).
        
        This method sets up the PSO algorithm with specified hyperparameters to optimize the parameters of a neural network,
        facilitating the training process by leveraging a population-based search strategy.
        
        Args:
            params (iterable): Iterable of parameters to optimize (e.g., model parameters).
            pop_size (int, optional): Population size of the PSO swarm. Defaults to 30.
            b (float, optional): Inertia weight controlling the influence of the particle's previous velocity. Defaults to 0.9.
            c1 (float, optional): Cognitive coefficient, weighing the influence of the particle's best known position. Defaults to 0.08.
            c2 (float, optional): Social coefficient, weighing the influence of the swarm's best known position. Defaults to 0.5.
            lr (float, optional): Learning rate for the optional gradient descent component. Defaults to 1e-3.
            betas (tuple(float, float), optional): Coefficients for computing running averages of gradient and its square (like in Adam). Defaults to (0.99, 0.999).
            c_decrease (bool, optional): Flag indicating whether to decrease cognitive and social coefficients over time. Defaults to False.
            variance (float, optional): Variance parameter used in the initialization of the swarm's positions. Defaults to 1.
            epsilon (float, optional): Small value added for numerical stability in the gradient descent update. Defaults to 1e-8.
            n_iter (int, optional): Number of iterations for the PSO algorithm. Defaults to 2000.
        
        Returns:
            None
        """
        defaults = {'pop_size': pop_size,
                    'b': b, 'c1': c1, 'c2': c2,
                    'lr': lr, 'betas': betas,
                    'c_decrease': c_decrease,
                    'variance': variance,
                    'epsilon': epsilon}
        super(PSO, self).__init__(params, defaults)
        self.params = self.param_groups[0]['params']
        self.pop_size = pop_size
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.c_decrease = c_decrease
        self.epsilon = epsilon
        self.beta1, self.beta2 = betas
        self.lr = lr * np.sqrt(1 - self.beta2) / (1 - self.beta1)
        self.use_grad = True if self.lr != 0 else False
        self.variance = variance
        self.name = "PSO"
        self.n_iter = n_iter

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
        
        This is a utility function to represent the model's state in a flattened format,
        which is useful for optimization algorithms that operate on vectors.
        
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
        Updates the model's parameters using a given vector.
        
        This method maps a vector from the optimization process
        onto the neural network's parameters, allowing the solver
        to explore different configurations during training.
        
        Args:
            vec (torch.Tensor): A vector representing a potential set of model parameters.
        
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
        
        The swarm is created by perturbing a base solution vector with random variance, generating a diverse set of candidate solutions.
        The first particle in the swarm is set to the original solution vector.
        
        Args:
            None
        
        Returns:
            torch.Tensor: A tensor representing the swarm population. Each row corresponds to a particle,
            representing a set of neural network parameters that approximate the solution to the differential equation.
            The tensor is detached from the computation graph and requires gradients for optimization.
        
        Why:
            Creating a swarm of solutions allows for exploration of the solution space,
            which is crucial for training a neural network to accurately solve the differential equation.
            The variance ensures diversity in the initial population, aiding in escaping local minima during training.
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

    def update_pso_params(self) -> None:
        """
        Updates the cognitive (c1) and social (c2) parameters of the PSO algorithm.
        
        This adjustment refines the balance between individual particle exploration and swarm influence
        during the optimization process, potentially leading to more accurate solutions of differential equations.
        The cognitive parameter (c1) is decreased, reducing the particle's reliance on its own past best position,
        while the social parameter (c2) is increased, enhancing the influence of the swarm's best position.
        
        Args:
            None
        
        Returns:
            None
        """
        self.c1 -= 2 * self.c1 / self.n_iter
        self.c2 += self.c2 / self.n_iter

    def start_velocities(self) -> torch.Tensor:
        """
        Initializes particle velocities to zero, ensuring a neutral starting point for optimization.
        
        This initialization is crucial for the optimization process, as it allows particles to explore the solution space without initial bias,
        facilitating a more comprehensive search for the optimal solution of the differential equation.
        
        Args:
            None
        
        Returns:
            torch.Tensor: A tensor of zeros representing the initial velocities of all particles in the swarm,
                          with shape (population size, vector shape).
        """
        return torch.zeros((self.pop_size, self.vec_shape))

    def gradient(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Calculation of the gradient of the loss with respect to the model parameters.
        
        This gradient is crucial for updating the particle's position during the optimization process, 
        allowing the swarm to explore the solution space of the differential equation.
        
        Args:
            loss (torch.Tensor): The calculated loss value, representing the error between the neural network's prediction and the true solution.
        
        Returns:
            torch.Tensor: A vector containing the gradient of the loss with respect to each model parameter.
        """
        dl_dparam = torch.autograd.grad(loss, self.params)

        grads = parameters_to_vector(dl_dparam)

        return grads

    def get_randoms(self) -> torch.Tensor:
        """
        Generate random values to update the particles' positions.
                This is needed to explore the solution space when training neural networks to solve differential equations.
        
                Returns:
                    torch.Tensor: random tensor
        """
        return torch.rand((2, 1, self.vec_shape))

    def update_p_best(self) -> None:
        """
        Updates the personal best positions of particles in the swarm.
        
        The personal best position of a particle is updated if the current position
        of the particle results in a lower loss than its previous personal best.
        This ensures that each particle remembers its best-performing location
        encountered so far during the optimization process, guiding the swarm
        towards better solutions of the differential equation.
        
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
        Update the global best position (*g-best*).
        
        The *g-best* represents the best solution found by the swarm so far. This method updates it by selecting the particle with the lowest function value among all particles' personal best positions. This ensures that the swarm converges towards the best solution discovered during the optimization process, which is crucial for finding accurate solutions to differential equations.
        
        Args:
            None
        
        Returns:
            None
        """
        self.g_best = self.p[torch.argmin(self.f_p)]

    def gradient_descent(self) -> torch.Tensor:
        """
        Updates velocities of particles using a gradient descent based on the Adam algorithm.
        
        This update refines the search trajectory of each particle by incorporating
        momentum and adaptive learning rates, which helps in navigating the solution
        space of the differential equation more effectively.
        
        Args:
            None
        
        Returns:
            torch.Tensor: The calculated update to the velocities vector.
        """
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * self.grads_swarm
        self.m2 = self.beta2 * self.m2 + (1 - self.beta2) * torch.square(
            self.grads_swarm)

        update = self.lr * self.m1 / (torch.sqrt(torch.abs(self.m2)) + self.epsilon)

        return update

    def step(self, closure=None) -> torch.Tensor:
        """
        Runs a single iteration of the Particle Swarm Optimization (PSO) algorithm to refine the solution of a differential equation represented by a neural network. It computes the loss and gradients, adjusts particle positions based on individual and global best solutions, and updates PSO parameters.
        
                Args:
                    closure (callable, optional): A function that evaluates the loss and gradients of the swarm. Defaults to None.
        
                Returns:
                    torch.Tensor: The minimum loss value achieved by the swarm, representing the best approximation to the differential equation's solution found in this iteration. This value is used to assess the quality of the current solution and guide further optimization.
        """

        self.loss_swarm, self.grads_swarm = closure()

        fix_attempt=0

        while torch.any(self.loss_swarm!=self.loss_swarm):
            self.swarm=self.swarm+0.001*torch.rand(size=self.swarm.shape)
            self.loss_swarm, self.grads_swarm = closure()
            fix_attempt+=1
            if fix_attempt>5:
                break

        if self.indicator:
            self.f_p = copy(self.loss_swarm).detach()
            self.g_best = self.p[torch.argmin(self.f_p)]
            self.indicator = False

        r1, r2 = self.get_randoms()

        self.v = self.b * self.v + (1 - self.b) * (
                self.c1 * r1 * (self.p - self.swarm) + self.c2 * r2 * (self.g_best - self.swarm))
        if self.use_grad:
            self.swarm = self.swarm + self.v - self.gradient_descent()
        else:
            self.swarm = self.swarm + self.v
        self.update_p_best()
        self.update_g_best()
        self.vec_to_params(self.g_best)
        if self.c_decrease:
            self.update_pso_params()
        min_loss = torch.min(self.f_p)

        return min_loss
