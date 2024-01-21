import torch
import numpy as np

from typing import Any
from copy import deepcopy

from tedeous.device import check_device, device_type

class Closure():
    def __init__(self,
        mixed_precision: bool,
        model):

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
        self._model = model

    @property
    def model(self):
        return self._model

    def _amp_mixed(self):
        """ Preparation for mixed precsion operations.

        Args:
            mixed_precision (bool): use or not torch.amp.

        Raises:
            NotImplementedError: AMP and the LBFGS optimizer are not compatible.

        Returns:
            scaler: GradScaler for CUDA.
            cuda_flag (bool): True, if CUDA is activated and mixed_precision=True.
            dtype (dtype): operations dtype.
        """

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        if self.mixed_precision:
            print(f'Mixed precision enabled. The device is {self.device}')
        if self.optimizer.__class__.__name__ == "LBFGS":
            raise NotImplementedError("AMP and the LBFGS optimizer are not compatible.")
        

    def _closure(self):
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
        def loss_grads():
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device,
                                dtype=self.dtype,
                                enabled=self.mixed_precision):
                loss, loss_normalized = self.model.solution_cls.evaluate()

            if self.optimizer.use_grad:
                grads = self.optimizer.gradient(loss)
                grads = torch.where(grads == float('nan'), torch.zeros_like(grads), grads)
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

    def get_closure(self, _type: str):
        if _type == 'PSO':
            return self._closure_pso
        else:
            return self._closure

    # def _closure_zo(self, size_params, mu, N_samples, input_size, d, sampler, gradient_mode):
    #     init_model_parameters = deepcopy(dict(self.model.net.state_dict()))
    #     model_parameters = dict(self.model.net.state_dict()).values()

    #     def parameter_perturbation(eps):
    #         start_idx = 0
    #         for param in model_parameters:
    #             end_idx = start_idx + param.view(-1).size()[0]
    #             param.add_(eps[start_idx: end_idx].view(param.size()).float(), alpha=np.sqrt(mu))
    #             start_idx = end_idx

    #     def grads_multiplication(grads, u):
    #         start_idx = 0
    #         grad_est = []
    #         for param in model_parameters:
    #             end_idx = start_idx + param.view(-1).size()[0]
    #             grad_est.append(grads * u[start_idx:end_idx].view(param.size()))
    #             start_idx = end_idx
    #         return grad_est

    #     grads = [torch.zeros_like(param) for param in model_parameters]
    #     self.cur_loss, _ = self.sln_cls.evaluate(second_order_interactions, sampling_N, lambda_update)

    #     for _ in range(N_samples):
    #         with torch.no_grad():
    #             if sampler == 'uniform':
    #                 u = 2 * (torch.rand(size_params) - 0.5)
    #                 u.div_(torch.norm(u, "fro"))
    #                 u = check_device(u)
    #             elif sampler == 'normal':
    #                 u = torch.randn(size_params)
    #                 u = check_device(u)

    #             # param + mu * eps
    #             parameter_perturbation(u)
    #         loss_add, _ = self.sln_cls.evaluate(second_order_interactions, sampling_N, lambda_update)

    #         # param - mu * eps
    #         with torch.no_grad():
    #             parameter_perturbation(-2 * u)
    #         loss_sub, _ = self.sln_cls.evaluate(second_order_interactions, sampling_N, lambda_update)

    #         with torch.no_grad():
    #             if gradient_mode == 'central':
    #                 # (1/ inp_size * q) * d * [f(x+mu*eps) - f(x-mu*eps)] / 2*mu
    #                 grad_coeff = (1 / (input_size * N_samples)) * d * (loss_add - loss_sub) / (2 * mu)
    #             elif gradient_mode == 'forward':
    #                 # d * [f(x+mu*eps) - f(x)] / mu
    #                 grad_coeff = (1 / (input_size * N_samples)) * d * (loss_add - self.cur_loss) / mu
    #             elif gradient_mode == 'backward':
    #                 # d * [f(x) - f(x-mu*eps)] / mu
    #                 grad_coeff = (1 / (input_size * N_samples)) * d * (self.cur_loss - loss_sub) / mu

    #             # coeff * u, i.e. constant multiplied by infinitely small perturbation.
    #             current_grad = grads_multiplication(grad_coeff, u)

    #             grads = [grad_past + cur_grad for grad_past, cur_grad in zip(grads, current_grad)]

    #         # load initial model parameters
    #         self.model.load_state_dict(init_model_parameters)

    #         loss_checker, _ = self.sln_cls.evaluate(second_order_interactions, sampling_N, lambda_update)
    #         assert self.cur_loss == loss_checker

    #     return grads