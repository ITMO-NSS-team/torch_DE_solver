import torch
import numpy as np
from copy import deepcopy
from points_type import Points_type
from derivative import Derivative
from device import device_type, check_device

flatten_list = lambda t: [item for sublist in t for item in sublist]


def integration(func, grid, pow=2): 
    """
    Function realize 1-space integrands,
    where func=(L(u)-f)*weak_form subintegrands function and
    definite integral parameter is grid
    Parameters:
    ----------
    func: torch.tensor
        operator mulilied on test function
    ----------
    grid: torch.tensor
    ----------
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
            U += (grid[i][-1] - grid[i-1][-1]).item()*\
                                        (func[i]**pow + func[i-1]**pow) / 2
        else:
            result.append(U)
            marker = grid[i][column]
            index.append(i)
            U = 0.
    if column == -1:
        return U, 0.
    else:
        result.append(U)
        grid = grid[index,:-1]
        return result, grid


class Solution():
    def __init__(self, grid, equal_cls, model, mode):
        self.grid = check_device(grid)
        equal_copy = deepcopy(equal_cls)
        self.prepared_operator = equal_copy.operator_prepare()
        self.prepared_bconds = equal_copy.bnd_prepare()
        self.model = model
        self.mode = mode
        if self.mode=='NN':
            self.grid_dict = Points_type(self.grid).grid_sort()
            self.sorted_grid = torch.cat(list(self.grid_dict.values()))
        elif self.mode=='autograd' or self.mode=='mat':
            self.sorted_grid = self.grid

    def apply_operator(self, operator, grid_points):
        """
        Deciphers equation in a single grid subset to a field.
        Parameters
        ----------
        model : torch.Sequential
            Neural network.
        operator : list
            Single (len(subset)==1) operator in input form. See 
            input_preprocessing.operator_prepare()
        Returns
        -------
        total : torch.Tensor
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

    def apply_bconds_set(self, operator_set):
        """
        Deciphers equation in a whole grid to a field.
        Parameters
        ----------
        model : torch.Sequential
            Neural network.
        operator : list
            Multiple (len(subset)>=1) operators in input form. See 
            input_preprocessing.operator_prepare()
        Returns
        -------
        total : torch.Tensor
        """
        field_part = []
        for operator in operator_set:
            field_part.append(self.apply_operator(operator, None))
        field_part = torch.cat(field_part)
        return field_part

    def apply_dirichlet(self, bnd, var):
        if self.mode == 'NN' or self.mode=='autograd':
            b_op_val = self.model(bnd)[:,var].reshape(-1,1)
        elif self.mode == 'mat':
            b_op_val=[]
            for position in bnd:
                if self.grid.dim() == 1 or min(self.grid.shape) == 1:
                    b_op_val.append(self.model[:, position])
                else:
                    b_op_val.append(self.model[position])
            b_op_val = torch.cat(b_op_val).reshape(-1,1)
        return b_op_val

    def apply_neumann(self, bnd, bop):
        if self.mode == 'NN':
            b_op_val = self.apply_bconds_set(bop)
        elif self.mode == 'autograd':
            b_op_val = self.apply_operator(bop, bnd)
        elif self.mode == 'mat':
            b_op_val = self.apply_operator(bop, self.grid)
            b_val=[]
            for position in bnd:
                if self.grid.dim() == 1 or min(self.grid.shape) == 1:
                    b_val.append(b_op_val[:, position])
                else:
                    b_val.append(b_op_val[position])
            b_op_val = torch.cat(b_val).reshape(-1,1)
        return b_op_val

    def apply_periodic(self, bnd, bop, var):

        if bop is None:
            b_op_val = self.apply_dirichlet(bnd[0], var).reshape(-1,1)
            for i in range(1, len(bnd)):
                b_op_val -= self.apply_dirichlet(bnd[i],var).reshape(-1,1)
        else:
            if self.mode == 'NN':
                b_op_val = self.apply_neumann(bnd, bop[0]).reshape(-1,1)
                for i in range(1, len(bop)):
                    b_op_val -= self.apply_neumann(bnd,bop[i]).reshape(-1,1)
            elif self.mode == 'autograd' or self.mode == 'mat':
                b_op_val = self.apply_neumann(bnd[0], bop).reshape(-1,1)
                for i in range(1, len(bnd)):
                    b_op_val -= self.apply_neumann(bnd[i],bop).reshape(-1,1)
        return b_op_val

    def b_op_val_calc(self, bcond):
        if bcond['type']=='dirichlet':
            b_op_val = self.apply_dirichlet(bcond['bnd'], bcond['var'])
        elif bcond['type']=='operator':
            b_op_val = self.apply_neumann(bcond['bnd'], bcond['bop'])
        elif bcond['type']=='periodic':
            b_op_val = self.apply_periodic(bcond['bnd'],bcond['bop'],
                                           bcond['var'])
        return b_op_val

    def apply_bconds_operator(self):
        true_b_val_list = []
        b_val_list = []

        for bcond in self.prepared_bconds:
            truebval = bcond['bval'].reshape(-1,1)

            true_b_val_list.append(truebval)

            b_op_val = self.b_op_val_calc(bcond)
            
            b_val_list.append(b_op_val)

        true_b_val = torch.cat(true_b_val_list)
        b_val = torch.cat(b_val_list).reshape(-1,1)

        return b_val, true_b_val

    def pde_compute(self):
        num_of_eq = len(self.prepared_operator)
        if num_of_eq == 1:
            op = self.apply_operator(
                self.prepared_operator[0], self.sorted_grid)
        else:
            op_list = []
            for i in range(num_of_eq):
                op_list.append(self.apply_operator(
                    self.prepared_operator[i], self.sorted_grid))
            op = torch.cat(op_list,1)
        return op

    def weak_pde_compute(self, weak_form):
        '''
        Weak solution of O/PDE problem.
        Parameters:
        ---------
        weak_form: list of basis functions
        lambda_bound: const regularization parameter
        ---------
        '''
        device = device_type()
        if self.mode=='NN':
            grid_central = self.grid_dict['central']
        elif self.mode=='autograd':
            grid_central = self.grid

        op = self.pde_compute()
        sol_list = []
        for i in range(op.shape[-1]):
            sol = op[:,i]
            for func in weak_form:
                sol = sol*func(grid_central).to(device).reshape(-1)
            grid_central1 = torch.clone(grid_central)
            for k in range(grid_central.shape[-1]):
                sol, grid_central1 = integration(sol, grid_central1)
            sol_list.append(sol.reshape(-1,1))
        if len(sol_list)==1:
            return sol_list[0]
        else:
            return torch.cat(sol_list)

    def l2_loss(self, lambda_bound=10):
        op = self.pde_compute()
        if self.prepared_bconds == None:
            return torch.sum(torch.mean((op) ** 2, 0))

        b_val, true_b_val = self.apply_bconds_operator()

        if self.mode=='mat':
            loss = torch.mean((op) ** 2) +\
            lambda_bound * torch.mean((b_val - true_b_val) ** 2)
        else:
            loss = torch.sum(torch.mean((op) ** 2, 0)) +\
            lambda_bound * torch.sum(torch.mean((b_val - true_b_val) ** 2, 0))
        return loss

    def weak_loss(self, weak_form, lambda_bound=10):
        op = self.weak_pde_compute(weak_form)
        if self.prepared_bconds == None:
            return torch.sum(op)

        # we apply no  boundary conditions operators if they are all None

        b_val, true_b_val = self.apply_bconds_operator()

        loss = torch.sum(op) +\
        lambda_bound * torch.sum(torch.mean((b_val - true_b_val) ** 2, 0))

        return loss

    def loss_evaluation(self, lambda_bound=10, weak_form=None):
        if self.mode == 'mat' or self.mode == 'autograd':
            if self.prepared_bconds == None:
                print('No bconds is not possible, returning infinite loss')
                return np.inf

        if weak_form == None or weak_form == []:
            return self.l2_loss(lambda_bound=lambda_bound)
        else:
            return self.weak_loss(weak_form, lambda_bound=lambda_bound)



