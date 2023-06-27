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
    """
    Serves for computing adaptive lambdas.

    Reference:
        S. Wang, X. Yu, and P. Perdikaris, “When and why PINNs fail to train: A neural tangent kernel
        perspective,” arXiv:2007.14527 [cs, math, stat], Jul. 2020, Available: https://arxiv.org/abs/2007.14527
    """
    def __init__(self, bounds: dict, operator, model):
        """
        Args:
            bounds: model output with boundary conditions at the input.
            operator: model output with operator at the input.
            model: NN

        """
        self.bnd = bounds
        self.op = operator
        self.model = model
        self.num_of_eq = operator.shape[-1]

    def jacobian(self, f: torch.Tensor) -> dict:
        """
        Computing Jacobian w.r.t model parameters (i.e. weights and biases).

        Args:
            f: function for which the Jacobian is calculated.

        Returns:
            Jacobian matrix.

        """
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

    @staticmethod
    def compute_ntk(J1_dict: dict, J2_dict: dict) -> torch.Tensor:
        """
        Computes neural tangent kernel.

        Args:
            J1_dict: dictionary with Jacobian.
            J2_dict: dictionary with Jacobian.

        Returns:
            NTK for corresponding input J1, J2.

        Reference:
            A. Jacot and F. Gabriel, “Neural Tangent Kernel: Convergence and Generalization in Neural Networks.” Available: https://arxiv.org/pdf/1806.07572.pdf
        """
        keys = list(J1_dict.keys())
        size = J1_dict[keys[0]].shape[0]
        Ker = torch.zeros((size, size))
        for key in keys:
            J1 = J1_dict[key]
            J2 = J2_dict[key]
            K = J1 @ J2.T
            Ker = Ker + K
        return Ker

    def trace(self, f: torch.Tensor) -> torch.Tensor:
        """
        Wrap all methods (Jacobian, NTK) and compute matrix trace.

        Args:
            f: function for which the trace is calculated.

        Returns:
            trace.

        """
        J_f = self.jacobian(f)
        ntk = self.compute_ntk(J_f, J_f)
        tr = torch.trace(ntk)
        return tr

    def update(self) -> dict:
        """
        Computes lambdas for corresponding boundary type.

        Returns:
            dictionary with corresponding lambdas.

        """
        traces_bcs = dict()
        lambda_bcs = dict()

        for bcs_type in self.bnd:
            traces_bcs[bcs_type] = self.trace(self.bnd[bcs_type])
            lambda_bcs[bcs_type] = []

        # if self.num_of_eq > 1:
        #     trace_op = torch.zeros(self.num_of_eq)
        #     for i in range(self.num_of_eq):
        #         trace_op[i] = self.trace(self.op[:, i: i + 1])
        #     trace_op = torch.mean(trace_op)
        # else:
        trace_op = self.trace(self.op)

        trace_K = trace_op + sum(traces_bcs.values())

        for bcs_type in traces_bcs:
            lambda_bcs[bcs_type] = trace_K / traces_bcs[bcs_type]

        return lambda_bcs