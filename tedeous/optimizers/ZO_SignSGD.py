import torch


class ZO_SignSGD(torch.optim.Optimizer):
    def __init__(self, params, input_size,
                 gradient_mode='central', sampler='normal',
                 n_samples=5, dim=2, lr=1e-3, mu=1e-3):

        defaults = dict(lr=lr, mu=mu)
        super().__init__(params, defaults)
        self.input_size = input_size
        self.gradient_mode = gradient_mode
        self.sampler = sampler
        self.n_samples = n_samples
        self.dim = dim
        self.name = 'ZO_SignSGD'

        self.size_params = 0
        for group in self.param_groups:
            for p in group['params']:
                self.size_params += torch.numel(p)

    def step(self, closure):
        for group in self.param_groups:
            lr = group['lr']
            for i, param in enumerate(group['params']):
                grad_est = closure(self.size_params, group["mu"],
                                   self.n_samples, self.input_size,
                                   self.dim, self.sampler, self.gradient_mode)

                param.data.add_(-lr * torch.sign(grad_est[i]))
