# this one contain some stuff for computing different auxiliary things.

import torch
from copy import copy

def list_to_vector(list_):
    return torch.cat([x.reshape(-1) for x in list_])


def counter(fu):
    def inner(*a, **kw):
        inner.count += 1
        return fu(*a, **kw)

    inner.count = 0
    return inner

class LambdaCompute():
    def __init__(self, bnd, operator, model):
        self.bval = bnd
        self.op = operator
        self.model = model
        self.num_of_eq = operator.shape[-1]

    def jacobian(self, f):
        jac = {}
        for name, param in self.model.named_parameters():
            jac1 = []
            for op in f:
                grad, = torch.autograd.grad(op, param, retain_graph=True, allow_unused=True)
                if grad is None:
                    grad = torch.tensor([0.])
                jac1.append(grad.reshape(1, -1))
            jac[name] = torch.cat(jac1)

        return jac

    def ntk(self, J1_dict, J2_dict):
        keys = list(J1_dict.keys())
        size = J1_dict[keys[0]].shape[0]
        Ker = torch.zeros((size, size))
        for key in keys:
            J1 = J1_dict[key]
            J2 = J2_dict[key]
            K = J1 @ J2.T
            Ker = Ker + K
        return Ker

    def trace(self, f):
        J_f = self.jacobian(f)
        ntk = self.ntk(J_f, J_f)
        tr = torch.trace(ntk)
        return tr

    def update(self):
        traces_bcs = dict()
        lambda_bcs = dict()

        for type in self.bval:
            traces_bcs[type] = self.trace(self.bval[type])
            lambda_bcs[type] = []

        if self.num_of_eq > 1:
            trace_op = torch.zeros(self.num_of_eq)
            for i in range(self.num_of_eq):
                 trace_op[i] = self.trace(self.op[:, i: i + 1])
            trace_op = torch.mean(trace_op)
        else:
            trace_op = torch.trace(self.op)

        trace_K = trace_op + sum(traces_bcs.values())

        for type in traces_bcs:
            lambda_bcs[type] = trace_K / traces_bcs[type]

        return lambda_bcs
