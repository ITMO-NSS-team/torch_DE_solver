# this one contain some stuff for computing different auxiliary things.

import torch

def list_to_vector(list_):
    return torch.cat([x.reshape(-1) for x in list_])

def counter(fu):
    def inner(*a,**kw):
        inner.count+=1
        return fu(*a,**kw)
    inner.count = 0
    return inner

class LambdaCompute():
    def __init__(self, bounds, bounds_op, operator, model):
        self.bnd = bounds
        self.bop = bounds_op
        self.op = operator
        self.model = model

    def jacobian(self, f):
        jac = {}
        for name, param in self.model.named_parameters():
            jac1 = []
            for op in f:
                grad, = torch.autograd.grad(op, param, retain_graph=True, allow_unused=True)
                if grad is None:
                    grad = torch.tensor([0.])
                jac1.append(grad.reshape(1,-1))
            jac[name] = torch.cat(jac1)

        return jac

    def compute_ntk(self, J1_dict, J2_dict):
        keys = list(J1_dict.keys())
        size = J1_dict[keys[0]].shape[0]
        Ker = torch.zeros((size, size))
        for key in keys:
            J1 = J1_dict[key]
            J2 = J2_dict[key]
            K = J1 @ J2.T
            Ker = Ker + K
        return Ker

    def update(self, iter, every):
        if iter % every == 0 or iter == -1:
            J_bnd = self.jacobian(self.bnd)
            J_bop = self.jacobian(self.bop)
            J_op = self.jacobian(self.op)

            K_bnd = self.compute_ntk(J_bnd, J_bnd)
            K_bop = self.compute_ntk(J_bop, J_bop)
            K_op = self.compute_ntk(J_op, J_op)

            trace_K = torch.trace(K_bnd) + torch.trace(K_bop) + \
                      torch.trace(K_op)

            l_bnd = trace_K / torch.trace(K_bnd)
            l_bop = trace_K / torch.trace(K_bop)
            l_op = trace_K / torch.trace(K_op)

            return l_bnd, l_bop, l_op