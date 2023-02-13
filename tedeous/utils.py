# this one contain some stuff for computing different auxiliary things.

import torch

def list_to_vector(list_):
    return torch.cat([x.reshape(-1) for x in list_])

class ComputeNTK():
    def __init__(self, bounds, operator, model):
        self.bounds = bounds
        self.operator = operator
        self.model = model

    def jacobian(self, function_):
        jac = []
        for point in function_:
            weights = self.model.parameters()
            jac.append(list_to_vector(torch.autograd.grad(point, weights, torch.ones_like(point), retain_graph=True, allow_unused=True)))
        jac = torch.vstack(jac)
        return jac

    def ntk(self):
        jacobian_bound, jacobian_operator = self.jacobian(self.bounds), self.jacobian(self.operator)
        Kuu = jacobian_bound @ jacobian_bound.T
        Krr = jacobian_operator @ jacobian_operator.T
        Kur = jacobian_bound @ jacobian_operator.T
        return Kuu, Krr, Kur

    def adapt_lambda(self):
        Kuu, Krr, Kur = self.ntk()
        K_trace = torch.trace(Kuu) + torch.trace(Krr)
        lambda_bound = K_trace / torch.trace(Kuu)
        lambda_operator = K_trace / torch.trace(Krr)
        lambda_ = lambda_bound / lambda_operator
        return lambda_