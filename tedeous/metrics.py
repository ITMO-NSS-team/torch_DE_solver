import torch
import numpy as np
from typing import Union, Any, Tuple

from tedeous.points_type import Points_type
from tedeous.input_preprocessing import *
from tedeous.utils import *
flatten_list = lambda t: [item for sublist in t for item in sublist]


class DerivativeInt():
    def take_derivative(self, value): 
        raise NotImplementedError

class Derivative_NN(DerivativeInt):
    def __init__(self, model: torch.nn.Sequential):
        """
        Taking numerical derivative for 'NN' method.
        Args:
             grid: array of a n-D points.
             model: neural network.
        """
        self.model = model
    
    def take_derivative (self, term: Union[list, int, torch.Tensor], *args) -> torch.Tensor:
        """
        Auxiliary function serves for single differential operator resulting field
        derivation.

        Args:
            term: differential operator in conventional form.
        Returns:
            resulting field, computed on a grid.
        """
        # it is may be int, function of grid or torch.Tensor
        if type(term[0]) is tuple:
            coeff = term[0][0](term[0][1]).reshape(-1,1)
        else:
            coeff = term[0]
        # this one contains shifted grids (see input_preprocessing module)
        shift_grid_list = term[1]
        # signs corresponding to a grid
        s_order_norm_list = term[2]
        # float that represents power of the differential term
        power = term[3]
        # number of variables in equation
        variables = term[4]
        # initially it is an ones field

        der_term = 1.
        for j, scheme in enumerate(shift_grid_list):
            # every shift in grid we should add with correspoiding sign, so we start
            # from zeros
            grid_sum = 0. 
            for k, grid in enumerate(scheme):
                # and add grid sequentially
                grid_sum += self.model(grid)[:,variables[j]].reshape(-1,1) * s_order_norm_list[j][k]
                # Here we want to apply differential operators for every term in the product
            der_term = der_term * grid_sum ** power[j]
        der_term = coeff * der_term
        return der_term

class Derivative_autograd(DerivativeInt):
    """
    Taking numerical derivative for 'autograd' method.
    """
    def __init__(self, model):
        self.model = model
    
    @staticmethod
    def nn_autograd(model: torch.nn.Sequential, points: torch.Tensor, var: int, axis: list = [0]) -> torch.Tensor :
        """
        Computes derivative on the grid using autograd method.
        Args:
            model: neural network.
            points: points, where numerical derivative is calculated.
            axis: smth
        Returns:
            smth
        """
        points.requires_grad=True
        fi = model(points)[:,var].sum(0)
        for ax in axis:
            grads, = torch.autograd.grad(fi, points, create_graph=True)
            fi = grads[:,ax].sum()
        gradient_full = grads[:,axis[-1]].reshape(-1,1)
        return gradient_full


    def take_derivative(self, term: Any, grid_points: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function serves for single differential operator resulting field
        derivation.

        Args:
            term: differential operator in conventional form.
        Returns:
            resulting field, computed on a grid.
        """
        # it is may be int, function of grid or torch.Tensor
        if callable(term[0]):
            coeff = term[0](grid_points).reshape(-1,1)
        else:
            coeff = term[0]
        # this one contains shifted grids (see input_preprocessing module)
        product = term[1]
        # float that represents power of the differential term
        power = term[2]
        # list that represent using variables
        variables = term[3]
        # initially it is an ones field
        der_term = 1.
        for j, derivative in enumerate(product):
            if derivative == [None]:
                der = self.model(grid_points)[:, variables[j]].reshape(-1, 1)
            else:
                der = self.nn_autograd(self.model, grid_points, variables[j], axis=derivative)
            der_term = der_term * der ** power[j]
        der_term = coeff * der_term
        return der_term

class Derivative_mat(DerivativeInt):
    def __init__(self, model: torch.Tensor):
        """
        Taking numerical derivative for 'mat' method.
        Args:
            model: random matrix.
        """
        self.model = model
    
    @staticmethod
    def derivative_1d(model: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        Computes derivative in one dimension for matrix method.
        Args:
            model: random matrix.
            grid: array of a n-D points.
        Returns:
            computed derivative along one dimension.
        """
        u=model.reshape(-1)
        x=grid.reshape(-1)
        
        # du_forward = (u-torch.roll(u, -1)) / (x-torch.roll(x, -1))
        
        # du_backward = (torch.roll(u, 1) - u) / (torch.roll(x, 1) - x)
        du =  (torch.roll(u, 1) - torch.roll(u, -1))/(torch.roll(x, 1)-torch.roll(x, -1))
        du[0] = (u[0]-u[1])/(x[0]-x[1])
        du[-1] = (u[-1]-u[-2])/(x[-1]-x[-2])
        
        du=du.reshape(model.shape)
        
        return du

    @staticmethod
    def derivative(u_tensor: torch.Tensor, h_tensor: torch.Tensor, axis: int,
                   scheme_order: int = 1, boundary_order: int = 1) -> torch.Tensor:
        """
        Computing derivative for 'matrix' method.
        Args:
            u_tensor: smth.
            h_tensor: smth.
            axis: axis along which the derivative is calculated.
            scheme_order: accuracy inner order for finite difference. Default = 1
            boundary_order: accuracy boundary order for finite difference. Default = 2
        Returns:
            computed derivative.
        """

        if (u_tensor.shape[0]==1):
            du = Derivative_mat.derivative_1d(u_tensor,h_tensor)
            return du

        u_tensor = torch.transpose(u_tensor, 0, axis)
        h_tensor = torch.transpose(h_tensor, 0, axis)
        
        
        if scheme_order==1:
            du_forward = (-torch.roll(u_tensor, -1) + u_tensor) / \
                        (-torch.roll(h_tensor, -1) + h_tensor)
        
            du_backward = (torch.roll(u_tensor, 1) - u_tensor) / \
                        (torch.roll(h_tensor, 1) - h_tensor)
            du = (1 / 2) * (du_forward + du_backward)
        
        # dh=h_tensor[0,1]-h_tensor[0,0]
        
        if scheme_order==2:
            u_shift_down_1 = torch.roll(u_tensor, 1)
            u_shift_down_2 = torch.roll(u_tensor, 2)
            u_shift_up_1 = torch.roll(u_tensor, -1)
            u_shift_up_2 = torch.roll(u_tensor, -2)
            
            h_shift_down_1 = torch.roll(h_tensor, 1)
            h_shift_down_2 = torch.roll(h_tensor, 2)
            h_shift_up_1 = torch.roll(h_tensor, -1)
            h_shift_up_2 = torch.roll(h_tensor, -2)
            
            h1_up=h_shift_up_1-h_tensor
            h2_up=h_shift_up_2-h_shift_up_1
            
            h1_down=h_tensor-h_shift_down_1
            h2_down=h_shift_down_1-h_shift_down_2
        
            a_up=-(2*h1_up+h2_up)/(h1_up*(h1_up+h2_up))
            b_up=(h2_up+h1_up)/(h1_up*h2_up)
            c_up=-h1_up/(h2_up*(h1_up+h2_up))
            
            a_down=(2*h1_down+h2_down)/(h1_down*(h1_down+h2_down))
            b_down=-(h2_down+h1_down)/(h1_down*h2_down)
            c_down=h1_down/(h2_down*(h1_down+h2_down))
            
            du_forward=a_up*u_tensor+b_up*u_shift_up_1+c_up*u_shift_up_2
            du_backward=a_down*u_tensor+b_down*u_shift_down_1+c_down*u_shift_down_2
            du = (1 / 2) * (du_forward + du_backward)
            
            
        if boundary_order==1:
            if scheme_order==1:
                du[:, 0] = du_forward[:, 0]
                du[:, -1] = du_backward[:, -1]
            elif scheme_order==2:
                du_forward = (-torch.roll(u_tensor, -1) + u_tensor) / \
                            (-torch.roll(h_tensor, -1) + h_tensor)
            
                du_backward = (torch.roll(u_tensor, 1) - u_tensor) / \
                            (torch.roll(h_tensor, 1) - h_tensor)
                du[:, 0] = du_forward[:, 0]
                du[:, 1] = du_forward[:, 1]
                du[:, -1] = du_backward[:, -1]
                du[:, -2] = du_backward[:, -2]
        elif boundary_order==2:
            if scheme_order==2:
                du[:, 0] = du_forward[:, 0]
                du[:, 1] = du_forward[:, 1]
                du[:, -1] = du_backward[:, -1]
                du[:, -2] = du_backward[:, -2]
            elif scheme_order==1:
                u_shift_down_1 = torch.roll(u_tensor, 1)
                u_shift_down_2 = torch.roll(u_tensor, 2)
                u_shift_up_1 = torch.roll(u_tensor, -1)
                u_shift_up_2 = torch.roll(u_tensor, -2)
                
                h_shift_down_1 = torch.roll(h_tensor, 1)
                h_shift_down_2 = torch.roll(h_tensor, 2)
                h_shift_up_1 = torch.roll(h_tensor, -1)
                h_shift_up_2 = torch.roll(h_tensor, -2)
                
                h1_up=h_shift_up_1-h_tensor
                h2_up=h_shift_up_2-h_shift_up_1
                
                h1_down=h_tensor-h_shift_down_1
                h2_down=h_shift_down_1-h_shift_down_2
            
        
                a_up=-(2*h1_up+h2_up)/(h1_up*(h1_up+h2_up))
                b_up=(h2_up+h1_up)/(h1_up*h2_up)
                c_up=-h1_up/(h2_up*(h1_up+h2_up))
                
                a_down=(2*h1_up+h2_up)/(h1_down*(h1_down+h2_down))
                b_down=-(h2_down+h1_down)/(h1_down*h2_down)
                c_down=h1_down/(h2_down*(h1_down+h2_down))
            
                
                du_forward=a_up*u_tensor+b_up*u_shift_up_1+c_up*u_shift_up_2
                du_backward=a_down*u_tensor+b_down*u_shift_down_1+c_down*u_shift_down_2
                du[:, 0] = du_forward[:, 0]
                du[:, -1] = du_backward[:, -1]
                
        du = torch.transpose(du, 0, axis)

        return du


    def take_derivative(self, term: Any, grid_points: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function serves for single differential operator resulting field
        derivation.

        Args:
            term: differential operator in conventional form.
            grid_points: grid points
        Returns:
            resulting field, computed on a grid.
        """

        # it is may be int, function of grid or torch.Tensor
        coeff = term[0]
        # this one contains product of differential operator
        operator_product = term[1]
        print(type(grid_points))
        # float that represents power of the differential term
        power = term[2]
        # initially it is an ones field
        der_term = torch.zeros_like(self.model) + 1
        for j, scheme in enumerate(operator_product):
            prod=self.model
            if scheme!=[None]:
                for axis in scheme:
                    if axis is None:
                        continue
                    h = grid_points[axis]
                    prod=self.derivative(prod, h, axis, scheme_order=1, boundary_order=1)
            der_term = der_term * prod ** power[j]
        if callable(coeff) is True:
            der_term = coeff(grid_points) * der_term
        else:
            der_term = coeff * der_term
        return der_term

class Derivative():
    def __init__(self, model):
        """
        Interface for taking numerical derivative due to chosen calculation method.

        Args:
            model: neural network or matrix depending on the selected mode.
        """
        self.model = model

    def set_strategy(self, strategy: str) -> Union[Derivative_NN, Derivative_autograd, Derivative_mat]:
        """
        Setting the calculation method.

        Args:
            strategy: Calculation method. (i.e., "NN", "autograd", "mat").
        Returns:
            equation in input form for a given calculation method.
        """

        if strategy == 'NN':
            return Derivative_NN(self.model)

        elif strategy == 'autograd':
            return  Derivative_autograd(self.model)

        elif strategy == 'mat':
            return Derivative_mat(self.model)


class Solution():
    def __init__(self, grid: torch.Tensor, equal_cls: Union[Equation_NN,
                 Equation_mat,Equation_autograd],
                 model: Union[torch.nn.Sequential, torch.Tensor], mode: str):
        self.grid = grid
        self.prepared_operator = equal_cls.operator_prepare()
        self.prepared_bconds = equal_cls.bnd_prepare()
        self.model = model
        self.mode = mode
        if self.mode=='NN':
            self.grid_dict = Points_type.grid_sort(self.grid)
            self.sorted_grid = torch.cat(list(self.grid_dict.values()))
        elif self.mode=='autograd' or self.mode=='mat':
            self.sorted_grid = self.grid

    def apply_operator(self, operator: list, grid_points) -> torch.Tensor:
        """
        Deciphers equation in a single grid subset to a field.
        Args:
            operator: Single (len(subset)==1) operator in input form. See
            input_preprocessing.operator_prepare()
            grid_points: grid points
        Returns:
            smth
        """
        derivative = Derivative(self.model).set_strategy(self.mode).take_derivative

        for term in operator:
            dif = derivative(term, grid_points)
            try:
                total += dif
            except NameError:
                total = dif
        return total

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
            field_part.append(self.apply_operator(operator, None))
        field_part = torch.cat(field_part)
        return field_part

    def b_op_val_calc(self, bcond: list) -> torch.Tensor:
        """
        Auxiliary function. Serves only to evaluate operator on the boundary.
        Args:
            bcond:  terms of prepared boundary conditions (see input_preprocessing.bnd_prepare) in input form.
        Returns:
            calculated operator on the boundary.
        """
        b_coord = bcond[0]
        bop = bcond[1]
        var = bcond[3]
        btype = bcond[4]

        if bop == None or bop == [[1, [None], 1]]:
            if self.mode == 'NN' or self.mode=='autograd':
                if btype=='boundary values':
                    b_op_val = self.model(b_coord)[:,var].reshape(-1,1)
                else:
                    b_op_val = self.model(b_coord[0])[:,var].reshape(-1,1)
                    for i in range(1, len(b_coord)):
                        b_op_val -= self.model(b_coord[i])[:,var].reshape(-1,1)

            elif self.mode == 'mat':
                b_op_val=[]
                for position in b_coord:
                    if self.grid.dim() == 1 or min(self.grid.shape) == 1:
                        b_op_val.append(self.model[:, position])
                    else:
                        b_op_val.append(self.model[position])
                b_op_val = torch.cat(b_op_val).reshape(-1,1)
        else:
            if self.mode == 'NN':
                if btype=='boundary values':
                    b_op_val = self.apply_bconds_set(bop)
                else:
                    b_op_val = self.apply_bconds_set(bop[0])
                    for i in range(1, len(bop)):
                        b_op_val -= self.apply_bconds_set(bop[i])
            elif self.mode == 'autograd':
                if btype=='boundary values':
                    b_op_val = self.apply_operator(bop, b_coord)
                else:
                    b_op_val = self.apply_operator(bop, b_coord[0])
                    for i in range(1, len(b_coord)):
                        b_op_val -= self.apply_operator(bop, b_coord[i])
            elif self.mode == 'mat':
                b_op_val = self.apply_operator(bop, self.grid)
                b_val=[]
                for position in b_coord:
                    if self.grid.dim() == 1 or min(self.grid.shape) == 1:
                        b_val.append(b_op_val[:, position])
                    else:
                        b_val.append(b_op_val[position])
                b_op_val = torch.cat(b_val).reshape(-1,1)

        return b_op_val

    def apply_bconds_operator(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Auxiliary function. Serves only to evaluate boundary values and true boundary values.
        Returns:
            * **b_val** -- calculated model boundary values.\n
            * **true_b_val** -- true grid boundary values.
        """
        true_b_val_list = []
        b_val_list = []

        # we apply no  boundary conditions operators if they are all None

        for bcond in self.prepared_bconds:
            truebval = bcond[2].reshape(-1,1)

            true_b_val_list.append(truebval)

            b_op_val = self.b_op_val_calc(bcond)
            
            b_val_list.append(b_op_val)

        true_b_val = torch.cat(true_b_val_list)
        b_val = torch.cat(b_val_list).reshape(-1,1)

        return b_val, true_b_val

    
    def l2_loss(self, lambda_bound:  Union[int, float] = 10, adaptive_lambda = False) -> torch.Tensor:
        """
        Computes l2 loss.
        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
        Returns:
            model loss.
        """
        if self.mode == 'mat' or self.mode == 'autograd':
            if self.prepared_bconds == None:
                print('No bconds is not possible, returning infinite loss')
                return np.inf

        num_of_eq = len(self.prepared_operator)
        if num_of_eq == 1:
            op = self.apply_operator(self.prepared_operator[0], self.sorted_grid)
            if self.prepared_bconds == None:
                return torch.mean((op) ** 2)
        else:
            op_list = []
            for i in range(num_of_eq):
                op_list.append(self.apply_operator(self.prepared_operator[i], self.sorted_grid))
            op = torch.cat(op_list,1)
            if self.prepared_bconds == None:
                return torch.sum(torch.mean((op) ** 2, 0))
    
        # we apply no  boundary conditions operators if they are all None

        b_val, true_b_val = self.apply_bconds_operator()
        if adaptive_lambda:
            lambda_bound = ComputeNTK(b_val - true_b_val, op, self.model).adapt_lambda()
        # print(lambda_)
        """
        actually, we can use L2 norm for the operator and L1 for boundary
        since L2>L1 and thus, boundary values become not so signifnicant, 
        so the NN converges faster. On the other hand - boundary conditions is the
        crucial thing for all that stuff, so we should increase significance of the
        coundary conditions
        """
        # l1_lambda = 0.001
        # l1_norm =sum(p.abs().sum() for p in model.parameters())
        # loss = torch.mean((op) ** 2) + lambda_bound * torch.mean((b_val - true_b_val) ** 2)+ l1_lambda * l1_norm
        if self.mode=='mat':
            loss = torch.mean((op) ** 2) + lambda_bound * torch.mean((b_val - true_b_val) ** 2)
        else:
            loss = torch.sum(torch.mean((op) ** 2, 0)) + lambda_bound * torch.sum(torch.mean((b_val - true_b_val) ** 2, 0))
        return loss

    def weak_loss(self, weak_form: Union[None, list], lambda_bound: Union[int, float] = 10) -> torch.Tensor:
        """
        Weak solution of O/PDE problem.
        Args:
            weak_form: list of basis functions.
            lambda_bound: const regularization parameter.
        Returns:
            model loss.
        """
        def integration(func: torch.Tensor, grid: torch.Tensor, pow: str ='sqrt'):
            """
            Function realize 1-space/multiple integrands, where func=(L(u)-f)*weak_form subintegrands function and
            definite integral parameter is grid.
            Args:
                func: basis function.
                grid: array of a n-D points.
                pow: (sqrt ar abs) power of func points.
            """
            if grid.shape[-1]==1:
                column = -1
            else:
                column = -2
            marker = grid[0][column]
            index = [0]
            result = []
            U = 0
            for i in range(1, len(grid)):
                if grid[i][column]==marker or column==-1:
                    if pow=='sqrt':
                        U += (grid[i][-1]-grid[i-1][-1]).item()*(func[i]**2+func[i-1]**2)/2
                    elif pow=='abs':
                        U += (grid[i][-1]-grid[i-1][-1]).item()*(func[i]+func[i-1])/2
                else:
                    result.append(U)
                    marker = grid[i][column]
                    index.append(i)
                    U = 0
            if column == -1:
                return U, 0
            else:
                result.append(U)
                grid = grid[index,:-1]
                return result, grid
        
        if self.mode=='NN':
            grid_central = self.grid_dict['central']
        elif self.mode=='autograd':
            grid_central = self.grid
        if self.mode == 'mat' or self.mode == 'autograd':
            if self.prepared_bconds == None:
                print('No bconds is not possible, returning infinite loss')
                return np.inf

        num_of_eq = len(self.prepared_operator)
        if num_of_eq == 1:
            op = self.apply_operator(self.prepared_operator[0], self.sorted_grid)
            for i in weak_form:
                op = op*(i(grid_central)).reshape(-1,1)
            for _ in range(grid_central.shape[-1]):
                op, grid_central = integration(op, grid_central)
            op_integr = op
            if self.prepared_bconds == None:
                return op_integr
        else:
            op_list = []
            for i in range(num_of_eq):
                op_list.append(self.apply_operator(self.prepared_operator[i], self.sorted_grid))
                for j in weak_form:
                    op_list[-1] = op_list[-1]*(j(grid_central)).reshape(-1,1)
            for i in range(num_of_eq):
                grid_central1 = torch.clone(grid_central)
                for k in range(grid_central.shape[-1]):
                    if k==0:
                        op_list[i], grid_central1 = integration(op_list[i], grid_central1, pow='sqrt')
                    else:
                        op_list[i], grid_central1 = integration(op_list[i], grid_central1, pow='sqrt')
            op_integr = torch.cat(op_list)
            if self.prepared_bconds == None:
                return op_integr

        # we apply no  boundary conditions operators if they are all None

        b_val, true_b_val = self.apply_bconds_operator()
        """
        actually, we can use L2 norm for the operator and L1 for boundary
        since L2>L1 and thus, boundary values become not so signifnicant, 
        so the NN converges faster. On the other hand - boundary conditions is the
        crucial thing for all that stuff, so we should increase significance of the
        coundary conditions
        """
        loss = torch.sum(op_integr) + lambda_bound * torch.sum(torch.mean((b_val - true_b_val) ** 2, 0))

        return loss

    def loss_evaluation(self, lambda_bound: Union[int, float] = 10, weak_form: Union[None, list] = None, adaptive_lambda = False) -> Union[l2_loss, weak_loss]:
        """
        Setting the required loss calculation method.
        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
            weak_form: list of basis functions.
        Returns:
            A given calculation method.
        """
        if weak_form == None or weak_form == []:
            return self.l2_loss(lambda_bound=lambda_bound, adaptive_lambda = adaptive_lambda)
        else:
            return self.weak_loss(weak_form, lambda_bound=lambda_bound)

