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
    def __init__(self, bounds, operator, model):
        self.bnd = bounds
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

    def trace(self, f):
        J_f = self.jacobian(f)
        ntk = self.compute_ntk(J_f, J_f)
        tr = torch.trace(ntk)
        return tr

    def update(self):
        traces_bcs = dict()
        lambda_bcs = dict()

        for bcs_type in self.bnd:
            traces_bcs[bcs_type] = self.trace(self.bnd[bcs_type])
            lambda_bcs[bcs_type] = []

        trace_op = self.trace(self.op)

        trace_K = trace_op + sum(traces_bcs.values())

        for bcs_type in traces_bcs:
            lambda_bcs[bcs_type] = trace_K / traces_bcs[bcs_type]

        return lambda_bcs