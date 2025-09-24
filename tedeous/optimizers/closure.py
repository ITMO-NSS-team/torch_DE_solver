import torch
from tedeous.device import device_type


class Closure():
    """
    A class that provides different closure methods for optimization.
    
        Class Methods:
        - __init__
        - set_model
        - model
        - _amp_mixed
        - _closure
        - _closure_pso
        - _closure_ngd
        - _closure_nncg
        - get_closure
    """

    def __init__(self,
                 mixed_precision: bool,
                 model
                 ):
        """
        Initializes the Trainer class, preparing the neural network model for solving differential equations.
        
                This method configures the training environment, including mixed precision settings, the model,
                optimizer, and device. This setup is crucial for efficiently training the neural network to approximate
                solutions to differential equations.
        
                Args:
                    mixed_precision (bool): Whether to use mixed precision training for faster and more memory-efficient computation.
                    model (torch.nn.Module): The neural network model to be trained for solving the differential equation.
        
                Returns:
                    None
        
                Class Fields:
                    mixed_precision (bool): A flag indicating whether mixed precision is enabled.
                    model (torch.nn.Module): The model to be trained.
                    optimizer (torch.optim.Optimizer): The optimizer used for training the model. Initialized from model.optimizer.
                    normalized_loss_stop (float): The threshold for normalized loss to stop training. Initialized from model.normalized_loss_stop.
                    device (str): The device type ('cuda' or 'cpu'). Determined by device_type().
                    cuda_flag (bool): A flag indicating whether CUDA is used with mixed precision.
                    dtype (torch.dtype): The data type used for training (torch.float16 if CUDA is used, otherwise torch.bfloat16).
        """

        self.mixed_precision = mixed_precision
        self.set_model(model)
        self.optimizer = self.model.optimizer
        self.normalized_loss_stop = self.model.normalized_loss_stop
        self.device = device_type()
        self.cuda_flag = True if self.device == 'cuda' and self.mixed_precision else False
        self.dtype = torch.float16 if self.device == 'cuda' else torch.bfloat16
        if self.mixed_precision:
            self._amp_mixed()

    def set_model(self, model):
        """
        Sets the underlying neural network model.
        
                This method is crucial for defining the architecture
                that will approximate the solution to the differential equation.
                By setting the model, you determine the function space
                within which the solution will be found.
        
                Args:
                    model (torch.nn.Module): The neural network model to be used
                        for approximating the solution.
        
                Returns:
                    None
        """
        self._model = model

    @property
    def model(self):
        """
        Gets the underlying neural network model.
        
                This property provides access to the neural network architecture
                used to approximate the solution of the differential equation.
                It allows inspection and modification of the model's parameters
                and structure, enabling customization and experimentation with
                different network designs for improved solution accuracy.
        
                Returns:
                    torch.nn.Module: The neural network model.
        """
        return self._model

    def _amp_mixed(self):
        """
        Prepares the environment for solving differential equations using mixed precision.
        
        This method configures the gradient scaler and data type based on whether mixed precision is enabled.
        It is crucial for leveraging the benefits of mixed precision, potentially accelerating the training process
        when solving differential equations with neural networks on CUDA-enabled devices.
        
        Args:
            mixed_precision (bool): A flag indicating whether to use mixed precision training.
        
        Raises:
            NotImplementedError: If the LBFGS optimizer is used with mixed precision, as they are incompatible.
        
        Returns:
            scaler (torch.cuda.amp.GradScaler): A gradient scaler instance for CUDA, enabled if mixed precision is True.
            cuda_flag (bool): True if CUDA is active and mixed_precision is True, False otherwise.
            dtype (torch.dtype): The data type to be used for operations (torch.float16 if mixed precision is enabled, torch.float32 otherwise).
        """

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        if self.mixed_precision:
            print(f'Mixed precision enabled. The device is {self.device}')
        if self.optimizer.__class__.__name__ == "LBFGS":
            raise NotImplementedError("AMP and the LBFGS optimizer are not compatible.")

    def _closure(self):
        """
        Performs a closure step for optimization, crucial for training the neural network to approximate the solution of the differential equation.
        
                This method encapsulates the core optimization loop: it zeroes gradients, evaluates the loss representing the error between the neural network's output and the differential equation, performs backpropagation to compute gradients, and updates the optimizer to adjust the network's weights. Mixed precision training is also handled here to improve computational efficiency. This entire process aims to minimize the discrepancy between the neural network's approximation and the true solution of the differential equation.
        
                Args:
                    self: The object instance.
        
                Returns:
                    torch.Tensor: The computed loss value, representing the error in the neural network's approximation of the differential equation's solution.
        """
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device,
                            dtype=self.dtype,
                            enabled=self.mixed_precision):
            loss, loss_normalized = self.model.solution_cls.evaluate()
        if self.cuda_flag:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()

        self.model.cur_loss = loss_normalized if self.normalized_loss_stop else loss

        return loss

    def _closure_pso(self):
        """
        Evaluates the loss and gradients for each particle in the swarm by training the neural network model.
        
                This method iterates through each particle, sets the model parameters
                according to the particle's position, and calculates the loss and gradients
                using the `loss_grads` inner function. This process effectively trains the
                neural network to approximate the solution of the differential equation.
                The losses and gradients for all particles are then collected and returned.
                This is done to find the best parameters (particle position) that minimize the loss,
                thereby improving the neural network's ability to solve the differential equation.
        
                Args:
                    self: The instance of the class containing this method.
        
                Returns:
                    tuple[torch.Tensor, torch.Tensor]: A tuple containing the losses and gradients for each particle in the swarm.
                        The first element is a tensor of losses, and the second element is a tensor of gradients.
        """
        def loss_grads():
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device,
                                dtype=self.dtype,
                                enabled=self.mixed_precision):
                loss, loss_normalized = self.model.solution_cls.evaluate()

            if self.optimizer.use_grad:
                grads = self.optimizer.gradient(loss)
                grads = torch.where(grads != grads, torch.zeros_like(grads), grads)
            else:
                grads = torch.tensor([0.])

            return loss, grads

        loss_swarm = []
        grads_swarm = []
        for particle in self.optimizer.swarm:
            self.optimizer.vec_to_params(particle)
            loss_particle, grads = loss_grads()
            loss_swarm.append(loss_particle)
            grads_swarm.append(grads.reshape(1, -1))

        losses = torch.stack(loss_swarm).reshape(-1)

        gradients = torch.vstack(grads_swarm)

        self.model.cur_loss = min(loss_swarm)

        return losses, gradients

    def _closure_ngd(self):
        """
        Performs a single optimization step using the NGD optimizer to refine the neural network's approximation of the differential equation's solution.
        
                This method computes the loss, performs backpropagation to adjust the network's parameters, and updates these parameters using the optimizer. Crucially, it also calculates the interior residual to assess how well the neural network satisfies the differential equation within the domain, and evaluates boundary conditions to ensure the solution adheres to the problem's constraints. This comprehensive evaluation guides the optimization process towards a more accurate solution.
        
                Args:
                    self: The object instance.
        
                Returns:
                    tuple: A tuple containing:
                           - int_res (torch.Tensor): The interior residual, quantifying the error within the domain.
                           - bval (torch.Tensor): The computed boundary values obtained from the neural network.
                           - true_bval (torch.Tensor): The true boundary values, representing the exact constraints.
                           - loss (torch.Tensor): The calculated loss, indicating the overall error in the solution.
                           - self.model.solution_cls.evaluate (callable): The evaluate function, used for subsequent evaluations of the solution.
        """
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device,
                            dtype=self.dtype,
                            enabled=self.mixed_precision):
            loss, loss_normalized = self.model.solution_cls.evaluate()
        if self.cuda_flag:
            self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward(retain_graph=True)

        self.model.cur_loss = loss_normalized if self.normalized_loss_stop else loss

        int_res = self.model.solution_cls.operator._pde_compute()
        bval, true_bval, _, _ = self.model.solution_cls.boundary.apply_bcs()

        return int_res, bval, true_bval, loss, self.model.solution_cls.evaluate

    def _closure_nncg(self):
        """
        Computes the loss and gradients by evaluating the neural network's approximation of the differential equation's solution.
        
                This method orchestrates the evaluation of the neural network model against the differential equation,
                calculating the loss that quantifies the discrepancy between the network's output and the true solution (or known constraints).
                It then computes the gradients of this loss with respect to the model's parameters, guiding the optimization process to refine the solution.
                Mixed precision training is supported to accelerate computations, and NaN gradients are checked and handled to ensure stability during training.
        
                Args:
                    self: The object instance.
        
                Returns:
                    tuple: A tuple containing the loss and the gradients.
                        - loss: The computed loss value, representing the error in the neural network's solution.
                        - grads: The gradients of the loss with respect to the model's parameters, used for optimization.
        """
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device,
                            dtype=self.dtype,
                            enabled=self.mixed_precision):
            loss, loss_normalized = self.model.solution_cls.evaluate()

        grads = self.optimizer.gradient(loss)
        grads = torch.where(grads != grads, torch.zeros_like(grads), grads)

        self.model.cur_loss = loss_normalized if self.normalized_loss_stop else loss

        return loss, grads

    def get_closure(self, _type: str):
        """
        Returns a specific optimization closure based on the provided type identifier.
        
                This function serves as a central point for selecting the appropriate optimization strategy
                when training neural networks to solve differential equations. Different closure types
                correspond to different optimization algorithms or strategies tailored for specific
                problem characteristics.
        
                Args:
                    _type (str): A string that identifies the type of closure to retrieve.
                                 Supported types include 'PSO', 'CSO', 'NGD', and 'NNCG', each
                                 representing a distinct optimization approach.
        
                Returns:
                    callable: The closure function corresponding to the given type. If the provided
                              type is not recognized, it returns a default closure, ensuring a
                              fallback optimization strategy is always available.
        """
        if _type in ('PSO', 'CSO'):
            return self._closure_pso
        elif _type == 'NGD':
            return self._closure_ngd
        elif _type == 'NNCG':
            return self._closure_nncg
        else:
            return self._closure
