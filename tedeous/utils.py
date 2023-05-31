# this one contain some stuff for computing different auxiliary things.

import torch


def list_to_vector(list_):
    return torch.cat([x.reshape(-1) for x in list_])


def counter(fu):
    def inner(*a, **kw):
        inner.count += 1
        return fu(*a, **kw)

    inner.count = 0
    return inner


class LambdaCompute():
    def __init__(self, bnd, true_bnd, operator, model):
        self.bnd = bnd[0]
        self.ics = bnd[1]
        self.true_bnd = true_bnd[0]
        self.true_ics = true_bnd[1]
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
        trace_bnd = self.trace(self.bnd)
        trace_ics = self.trace(self.ics)

        if self.num_of_eq > 1:
            trace_op = torch.zeros(self.num_of_eq)
            for i in range(self.num_of_eq):
                 trace_op[i] = self.trace(self.op[:, i: i + 1])
            trace_op = torch.mean(trace_op)
        else:
            trace_op = torch.trace(self.op)

        trace_K = trace_bnd + trace_ics + trace_op

        l_bnd = trace_K / trace_bnd
        l_ics = trace_K / trace_ics
        l_op = trace_K / trace_op

        return l_bnd, l_ics, l_op
