import torch
import numpy as np

from copy import deepcopy
from tedeous.device import check_device

class ZO_AdaMM(torch.optim.Optimizer):
    def __init__(self, params, input_size,
                 gradient_mode='forward', sampler='uniform',
                 dim=2, lr=1e-3, betas=(0.9, 0.999),
                 mu=1e-3, eps=1e-12):

        defaults = dict(lr=lr, betas=betas, mu=mu, eps=eps)
        super().__init__(params, defaults)
        self.input_size = input_size
        self.gradient_mode = gradient_mode
        self.sampler = sampler
        self.n_samples = 1
        self.dim = dim
        self.name = 'ZO_Adam'

        self.size_params = 0
        for group in self.param_groups:
            for p in group['params']:
                self.size_params += torch.numel(p)

    @staticmethod
    def closure(size_params, mu, N_samples, input_size, d, sampler, gradient_mode):
        init_model_parameters = deepcopy(dict(self.model.state_dict()))
        model_parameters = dict(self.model.state_dict()).values()

        def parameter_perturbation(eps):
            start_idx = 0
            for param in model_parameters:
                end_idx = start_idx + param.view(-1).size()[0]
                param.add_(eps[start_idx: end_idx].view(param.size()).float(), alpha=np.sqrt(mu))
                start_idx = end_idx

        def grads_multiplication(grads, u):
            start_idx = 0
            grad_est = []
            for param in model_parameters:
                end_idx = start_idx + param.view(-1).size()[0]
                grad_est.append(grads * u[start_idx:end_idx].view(param.size()))
                start_idx = end_idx
            return grad_est

        grads = [torch.zeros_like(param) for param in model_parameters]
        self.cur_loss, _ = self.sln_cls.evaluate(second_order_interactions, sampling_N, lambda_update)

        for _ in range(N_samples):
            with torch.no_grad():
                if sampler == 'uniform':
                    u = 2 * (torch.rand(size_params) - 0.5)
                    u.div_(torch.norm(u, "fro"))
                    u = check_device(u)
                elif sampler == 'normal':
                    u = torch.randn(size_params)
                    u = check_device(u)

                # param + mu * eps
                parameter_perturbation(u)
            loss_add, _ = self.sln_cls.evaluate(second_order_interactions, sampling_N, lambda_update)

            # param - mu * eps
            with torch.no_grad():
                parameter_perturbation(-2 * u)
            loss_sub, _ = self.sln_cls.evaluate(second_order_interactions, sampling_N, lambda_update)

            with torch.no_grad():
                if gradient_mode == 'central':
                    # (1/ inp_size * q) * d * [f(x+mu*eps) - f(x-mu*eps)] / 2*mu
                    grad_coeff = (1 / (input_size * N_samples)) * d * (loss_add - loss_sub) / (2 * mu)
                elif gradient_mode == 'forward':
                    # d * [f(x+mu*eps) - f(x)] / mu
                    grad_coeff = (1 / (input_size * N_samples)) * d * (loss_add - self.cur_loss) / mu
                elif gradient_mode == 'backward':
                    # d * [f(x) - f(x-mu*eps)] / mu
                    grad_coeff = (1 / (input_size * N_samples)) * d * (self.cur_loss - loss_sub) / mu

                # coeff * u, i.e. constant multiplied by infinitely small perturbation.
                current_grad = grads_multiplication(grad_coeff, u)

                grads = [grad_past + cur_grad for grad_past, cur_grad in zip(grads, current_grad)]

            # load initial model parameters
            self.model.load_state_dict(init_model_parameters)

            loss_checker, _ = self.sln_cls.evaluate(second_order_interactions, sampling_N, lambda_update)
            assert self.cur_loss == loss_checker

        return grads


    def step(self, closure):

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            # Closure return the approximation for the gradient
            grad_est = closure(self.size_params, group["mu"],
                               self.n_samples, self.input_size,
                               self.dim, self.sampler, self.gradient_mode)

            for p, grad in zip(group['params'], grad_est):
                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Do the AdaMM updates
                state['exp_avg'].mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))
                state['max_exp_avg_sq'] = torch.maximum(state['max_exp_avg_sq'],
                                                        state['exp_avg_sq'])

                p.data.addcdiv_(state['exp_avg'], state['exp_avg_sq'].sqrt().add_(group['eps']), value=(-group['lr']))
