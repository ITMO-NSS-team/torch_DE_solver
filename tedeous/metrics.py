import torch
import numpy as np
from copy import deepcopy
from typing import Tuple, Union

from tedeous.points_type import Points_type
from tedeous.derivative import Derivative
from tedeous.device import device_type, check_device
import tedeous.input_preprocessing
from tedeous.utils import *

flatten_list = lambda t: [item for sublist in t for item in sublist]


def integration(func: torch.tensor, grid, pow: Union[int, float] = 2) -> Union[Tuple[float, float], Tuple[list, torch.Tensor]]:
    """
    Function realize 1-space integrands,
    where func=(L(u)-f)*weak_form subintegrands function and
    definite integral parameter is grid.

    Args:
        func: operator multiplied on test function
        grid: array of a n-D points.
        pow: string (sqr ar abs) power of func points

    Returns:
        tuple(result, grid)
        'result' is integration result through one grid axis
        'grid' is initial grid without last column or zero (if grid.shape[N,1])
    """
    if grid.shape[-1] == 1:
        column = -1
    else:
        column = -2
    marker = grid[0][column]
    index = [0]
    result = []
    U = 0.
    for i in range(1, len(grid)):
        if grid[i][column] == marker or column == -1:
            U += (grid[i][-1] - grid[i - 1][-1]).item() * \
                 (func[i] ** pow + func[i - 1] ** pow) / 2
        else:
            result.append(U)
            marker = grid[i][column]
            index.append(i)
            U = 0.
    if column == -1:
        return U, 0.
    else:
        result.append(U)
        grid = grid[index, :-1]
        return result, grid

class Operator():
    def __init__(self,operator, grid, model, mode):
        self.operator = operator
        self.grid = grid
        self.model = model
        self.mode = mode

    def apply_op(self, operator, grid_points) -> torch.Tensor:
        """
        Deciphers equation in a single grid subset to a field.
        Args:
            operator: Single (len(subset)==1) operator in input form. See
            input_preprocessing.operator_prepare()
            grid_points: grid points
        Returns:
            smth
        """
        derivative = Derivative(self.model).set_strategy(
            self.mode).take_derivative

        for term in operator:
            term = operator[term]
            dif = derivative(term, grid_points)
            try:
                total += dif
            except NameError:
                total = dif
        return total

class Bounds(Operator):
    def __init__(self, bconds, grid, model, mode):
        super().__init__(bconds, grid, model, mode)
        self.bconds = bconds

    def apply_bconds_set(self, operator_set: list) -> torch.Tensor:
        """
        Deciphers equation in a whole grid to a field.
        Args:
            operator_set: Multiple (len(subset)>=1) operators in input form. See
            input_preprocessing.operator_prepare().
        Returns:
            smth
        """
        field_part = []
        for operator in operator_set:
            field_part.append(self.apply_op(operator, None))
        field_part = torch.cat(field_part)
        return field_part

    def apply_dirichlet(self, bnd, var):
        if self.mode == 'NN' or self.mode == 'autograd':
            b_op_val = self.model(bnd)[:, var].reshape(-1, 1)
        elif self.mode == 'mat':
            b_op_val = []
            for position in bnd:
                if self.grid.dim() == 1 or min(self.grid.shape) == 1:
                    b_op_val.append(self.model[:, position])
                else:
                    b_op_val.append(self.model[position])
            b_op_val = torch.cat(b_op_val).reshape(-1, 1)
        return b_op_val

    def apply_neumann(self, bnd, bop):
        if self.mode == 'NN':
            b_op_val = self.apply_bconds_set(bop)
        elif self.mode == 'autograd':
            b_op_val = self.apply_op(bop, bnd)
        elif self.mode == 'mat':
            b_op_val = self.apply_op(operator = bop, grid_points=self.grid)
            b_val = []
            for position in bnd:
                if self.grid.dim() == 1 or min(self.grid.shape) == 1:
                    b_val.append(b_op_val[:, position])
                else:
                    b_val.append(b_op_val[position])
            b_op_val = torch.cat(b_val).reshape(-1, 1)
        return b_op_val

    def apply_periodic(self, bnd, bop, var):
        if bop is None:
            b_op_val = self.apply_dirichlet(bnd[0], var).reshape(-1, 1)
            for i in range(1, len(bnd)):
                b_op_val -= self.apply_dirichlet(bnd[i], var).reshape(-1, 1)
        else:
            if self.mode == 'NN':
                b_op_val = self.apply_neumann(bnd, bop[0]).reshape(-1, 1)
                for i in range(1, len(bop)):
                    b_op_val -= self.apply_neumann(bnd, bop[i]).reshape(-1, 1)
            elif self.mode == 'autograd' or self.mode == 'mat':
                b_op_val = self.apply_neumann(bnd[0], bop).reshape(-1, 1)
                for i in range(1, len(bnd)):
                    b_op_val -= self.apply_neumann(bnd[i], bop).reshape(-1, 1)
        return b_op_val

    def b_op_val_calc(self, bcond) -> torch.Tensor:
        """
        Auxiliary function. Serves only to evaluate operator on the boundary.
        Args:
            bcond:  terms of prepared boundary conditions (see input_preprocessing.bnd_prepare) in input form.
        Returns:
            calculated operator on the boundary.
        """
        if bcond['type'] == 'dirichlet':
            b_op_val = self.apply_dirichlet(bcond['bnd'], bcond['var'])
        elif bcond['type'] == 'operator':
            b_op_val = self.apply_neumann(bcond['bnd'], bcond['bop'])
        elif bcond['type'] == 'periodic':
            b_op_val = self.apply_periodic(bcond['bnd'], bcond['bop'],
                                           bcond['var'])
        return b_op_val

    def compute_bconds(self, bcond):
        truebval = bcond['bval'].reshape(-1, 1)
        b_op_val = self.b_op_val_calc(bcond)
        return b_op_val, truebval
    def apply_bconds_default(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Auxiliary function. Serves only to evaluate boundary values and true boundary values.
        Returns:
            * **b_val** -- calculated model boundary values.\n
            * **true_b_val** -- true grid boundary values.
        """

        true_b_val_list = []
        b_val_list = []
        for bcond in self.bconds:
            b_op_val, truebval = self.compute_bconds(bcond)
            b_val_list.append(b_op_val)
            true_b_val_list.append(truebval)

        true_b_val = torch.cat(true_b_val_list)
        b_val = torch.cat(b_val_list).reshape(-1, 1)
        return b_val, true_b_val

    def apply_bconds_separate(self):
        true_b_val_list_sep = []
        b_val_list_sep = []

        true_b_val_list_op = []
        b_val_list_op = []
        for bcond in self.bconds:
            if bcond['type'] == 'operator':
                b_val_op, true_b_val_op = self.compute_bconds(bcond)
                b_val_list_op.append(b_val_op)
                true_b_val_list_op.append(true_b_val_op)
            else:
                b_val_sep, true_b_val_sep = self.compute_bconds(bcond)
                b_val_list_sep.append(b_val_sep)
                true_b_val_list_sep.append(true_b_val_sep)

        true_b_val_separate = torch.cat(true_b_val_list_sep)
        b_val_op_separate = torch.cat(b_val_list_sep)
        true_b_val_operator = torch.cat(true_b_val_list_op)
        b_val_op_operator = torch.cat(b_val_list_op)

        b_val = [b_val_op_separate, b_val_op_operator]
        true_b_val = [true_b_val_separate, true_b_val_operator]
        return b_val, true_b_val
    # def apply_bconds_separate(self):
    #     loss_bcs = []
    #     loss_ics = []
    #     for bcond in self.bconds:
    #         if bcond['type'] == 'operator':
    #             b_val_op, true_b_val_op = self.compute_bconds(bcond)
    #             loss_ics.append(torch.mean(torch.square(b_val_op)))
    #             true_b_val_list_op.append(true_b_val_op)
    #         else:
    #             b_val_sep, true_b_val_sep = self.compute_bconds(bcond)
    #             b_val_list_sep.append(b_val_sep)
    #             true_b_val_list_sep.append(true_b_val_sep)
    def apply_bnd(self, mode):
        if type(mode) is int:
            return self.apply_bconds_separate()
        else:
            return self.apply_bconds_default()


class Loss():
    def __init__(self, *args, operator, bval, true_bval, mode, weak_form):
        if len(args) == 3:
            self.l_bnd, self.l_bop, self.l_op = args[0], args[1], args[2]
        self.operator = operator
        self.bval = bval
        self.true_bval = true_bval
        self.mode = mode
        self.weak_form = weak_form




    def l2_loss(self, lambda_bound:  Union[int, float] = 10) -> torch.Tensor:
        """
        Computes l2 loss.
        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
        Returns:
            model loss.
        """
        if self.bval == None:
            return torch.sum(torch.mean((self.operator) ** 2, 0))

        if self.mode == 'mat':
            loss = torch.mean((self.operator) ** 2) + \
                   lambda_bound * torch.mean((self.bval - self.true_bval) ** 2)
        else:
            loss = torch.sum(torch.mean((self.operator) ** 2, 0)) + \
                   lambda_bound * torch.sum(torch.mean((self.bval - self.true_bval) ** 2, 0))
        return loss

    def weak_loss(self, lambda_bound: Union[int, float] = 10) -> torch.Tensor:
        """
        Weak solution of O/PDE problem.
        Args:
            weak_form: list of basis functions.
            lambda_bound: const regularization parameter.
        Returns:
            model loss.
        """
        if self.bval == None:
            return torch.sum(self.operator)

        # we apply no  boundary conditions operators if they are all None


        loss = torch.sum(self.operator) + \
               lambda_bound * torch.sum(torch.mean((self.bval - self.true_bval) ** 2, 0))

        return loss

    def l2_loss_sep(self, lambda_bound, lambda_bound_op, lambda_op) -> torch.Tensor:
        """
        Args:

        Returns:

        """

        if self.bval == None:
            return torch.sum(torch.mean((self.operator) ** 2, 0))

        loss = lambda_op * torch.sum(torch.mean((self.operator) ** 2), 0) + \
               lambda_bound * torch.sum(torch.mean((self.bval[0] - self.true_bval[0]) ** 2, 0)) + \
               lambda_bound_op * torch.sum(torch.mean((self.bval[1] - self.true_bval[1]) ** 2, 0))
        return loss

    def compute(self, *args, lambda_bound = 10) -> Union[l2_loss, weak_loss]:
        """
        Setting the required loss calculation method.
        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
            weak_form: list of basis functions.
        Returns:
            A given calculation method.
        """


        if self.mode == 'mat' or self.mode == 'autograd':
            if self.bval == None:
                print('No bconds is not possible, returning infinite loss')
                return np.inf

        if self.weak_form == None or self.weak_form == []:
            if len(args) == 3:
                return self.l2_loss_sep(self.l_bnd, self.l_bop, self.l_op)
            else:
                return self.l2_loss(lambda_bound=lambda_bound)
        else:
            return self.weak_loss(lambda_bound=lambda_bound)

class Solution():
    def __init__(self, grid: torch.Tensor, equal_cls: Union[tedeous.input_preprocessing.Equation_NN,
                                                            tedeous.input_preprocessing.Equation_mat, tedeous.input_preprocessing.Equation_autograd],
                 model: Union[torch.nn.Sequential, torch.Tensor], mode: str, weak_form, update_every_lambdas):
        self.grid = check_device(grid)
        equal_copy = deepcopy(equal_cls)
        self.prepared_operator = equal_copy.operator_prepare()
        self.prepared_bconds = equal_copy.bnd_prepare()
        self.model = model.to(device_type())
        self.mode = mode
        self.update_every_lambdas = update_every_lambdas
        self.weak_form = weak_form
        if self.mode == 'NN':
            self.grid_dict = Points_type(self.grid).grid_sort()
            self.sorted_grid = torch.cat(list(self.grid_dict.values()))
        elif self.mode == 'autograd' or self.mode == 'mat':
            self.sorted_grid = self.grid
        self.operator = Operator(operator = self.prepared_operator, grid = self.grid, model = self.model, mode = self.mode)
        self.l_bnd, self.l_bop, self.l_op = 1, 1, 1

    def pde_compute(self) -> torch.Tensor:
        """
        Computes PDE residual.

        Returns:
            PDE residual.
        """
        num_of_eq = len(self.prepared_operator)
        if num_of_eq == 1:
            op = self.operator.apply_op(
                self.prepared_operator[0], self.sorted_grid)
        else:
            op_list = []
            for i in range(num_of_eq):
                op_list.append(self.operator.apply_op(
                    self.prepared_operator[i], self.sorted_grid))
            op = torch.cat(op_list, 1)
        return op

    def weak_pde_compute(self, weak_form) -> torch.Tensor:
        """
        Computes PDE residual in weak form.

        Args:
            weak_form: list of basis functions
        Returns:
            weak PDE residual.
        """
        device = device_type()
        if self.mode == 'NN':
            grid_central = self.grid_dict['central']
        elif self.mode == 'autograd':
            grid_central = self.grid

        op = self.pde_compute()
        sol_list = []
        for i in range(op.shape[-1]):
            sol = op[:, i]
            for func in weak_form:
                sol = sol * func(grid_central).to(device).reshape(-1)
            grid_central1 = torch.clone(grid_central)
            for k in range(grid_central.shape[-1]):
                sol, grid_central1 = integration(sol, grid_central1)
            sol_list.append(sol.reshape(-1, 1))
        if len(sol_list) == 1:
            return sol_list[0]
        else:
            return torch.cat(sol_list)
    @counter
    def operator_compute(self):
        if self.weak_form == None or self.weak_form == []:
            return self.pde_compute()
        else:
            return self.weak_pde_compute(self.weak_form)


    def evaluate(self, lambda_bound):
        it = self.operator_compute.count
        op = self.operator_compute()
        bval, true_bval = Bounds(self.prepared_bconds,self.grid, self.model,self.mode).apply_bnd(mode=self.update_every_lambdas)
        loss = Loss(operator=op,bval=bval,true_bval=true_bval,mode=self.mode, weak_form = self.weak_form)

        if type(self.update_every_lambdas) is int:
            bnd = bval[0]
            true_bnd = true_bval[0]

            bop = bval[1]
            true_bop = true_bval[1]

            lambdas = LambdaCompute(bnd - true_bnd, bop, op, self.model)
            loss = Loss(self.l_bnd, self.l_bop, self.l_op, operator=op,bval=bval,true_bval=true_bval,mode=self.mode, weak_form = self.weak_form)
            if it % self.update_every_lambdas == 0:
                self.l_bnd, self.l_bop, self.l_op = lambdas.update()
            return loss.compute(self.l_bnd, self.l_bop, self.l_op)

        return loss.compute(lambda_bound)