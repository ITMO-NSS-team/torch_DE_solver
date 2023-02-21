import torch
import numpy as np



class DerivativeInt():
    def take_derivative(self, value):
        raise NotImplementedError


class Derivative_NN(DerivativeInt):
    """
    Class for taking derivatives by the 'NN' method.

    Parameters:
    -----------
    model: torch.nn.Sequential
        neural network

    """

    def __init__(self, model):
        self.model = model

    def take_derivative(self, term, *args):
        """
        Nethod serves for single operator differentiation

        Parameters
        ----------
        term : dict
            differential operator in conventional form.
        Returns
        -------
        der_term : torch.Tensor
            resulting field, computed on a grid.

        """

        dif_dir = list(term.keys())[1]
        if type(term['coeff']) is tuple:
            coeff = term['coeff'][0](term['coeff'][1]).reshape(-1, 1)
        else:
            coeff = term['coeff']

        der_term = 1.
        for j, scheme in enumerate(term[dif_dir][0]):
            grid_sum = 0.
            for k, grid in enumerate(scheme):
                grid_sum += self.model(grid)[:, term['var'][j]].reshape(-1, 1)\
                    * term[dif_dir][1][j][k]
            der_term = der_term * grid_sum ** term['pow'][j]
        der_term = coeff * der_term

        return der_term


class Derivative_autograd(DerivativeInt):
    """
    Class for taking derivatives by the 'autograd' method.

    Parameters:
    -----------
    model: torch.nn.Sequential
        neural network

    """

    def __init__(self, model):
        self.model = model

    @staticmethod
    def nn_autograd(model, points, var, axis=[0]):
        """
        Static Method for autograd differentiation

        Parameters:
        -----------
        model: torch.nn.Module
            Neural network
        -----------
        points: torch.Tensor
            grid points where gradient needeed
        -----------
        var: int
            in system case, var is number of desired function
        ----------
        axis: list
            term of differentiation, example [0,0]->d2/dx2
            if grid_points(x,y)

        Returns:
        ----------
            the result of desired function deifferentiation
            in corrsponding axis

        """

        points.requires_grad = True
        fi = model(points)[:, var].sum(0)
        for ax in axis:
            grads, = torch.autograd.grad(fi, points, create_graph=True)
            fi = grads[:, ax].sum()
        gradient_full = grads[:, axis[-1]].reshape(-1, 1)
        return gradient_full

    def take_derivative(self, term, grid_points):
        """
        Axiluary function serves for single differential operator resulting
        field derivation.

        Parameters
        ----------
        model : torch.Sequential
            Neural network.
        term : dict
            differential operator.
        Returns
        -------
        der_term : torch.Tensor
            resulting field, computed on a grid.

        """

        dif_dir = list(term.keys())[1]
        # it is may be int, function of grid or torch.Tensor
        if callable(term['coeff']):
            coeff = term['coeff'](grid_points).reshape(-1, 1)
        else:
            coeff = term['coeff']

        der_term = 1.
        for j, derivative in enumerate(term[dif_dir]):
            if derivative == [None]:
                der = self.model(grid_points)[:, term['var'][j]].reshape(-1, 1)
            else:
                der = self.nn_autograd(
                    self.model, grid_points, term['var'][j], axis=derivative)
            der_term = der_term * der ** term['pow'][j]
        der_term = coeff * der_term

        return der_term


class Derivative_mat(DerivativeInt):
    """
    Class for taking derivatives by the 'mat' method.

    Parameters:
    -----------
    model: torch.Tensor

    """

    def __init__(self, model):
        self.model = model

    @staticmethod
    def derivative_1d(model, grid):
        # print('1d>2d')
        u = model.reshape(-1)
        x = grid.reshape(-1)

        # du_forward = (u-torch.roll(u, -1)) / (x-torch.roll(x, -1))
        # du_backward = (torch.roll(u, 1) - u) / (torch.roll(x, 1) - x)

        du = (torch.roll(u, 1) - torch.roll(u, -1))/\
                                        (torch.roll(x, 1) - torch.roll(x, -1))
        du[0] = (u[0] - u[1]) / (x[0] - x[1])
        du[-1] = (u[-1] - u[-2]) / (x[-1] - x[-2])

        du=du.reshape(model.shape)

        return du

    @staticmethod
    def derivative(u_tensor, h_tensor, axis, scheme_order=1, boundary_order=1):
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

    def take_derivative(self, term, grid_points):
        """
        Axiluary function serves for single differential operator resulting
        field derivation.

        Parameters
        ----------
        model : torch.tensor

        term : dict
            differential operator.

        Returns
        -------
        der_term : torch.Tensor
            resulting field, computed on a grid.

        """

        dif_dir = list(term.keys())[1]
        der_term = torch.zeros_like(self.model) + 1
        for j, scheme in enumerate(term[dif_dir]):
            prod=self.model
            if scheme!=[None]:
                for axis in scheme:
                    if axis is None:
                        continue
                    h = grid_points[axis]
                    prod=self.derivative(prod, h, axis, scheme_order=1, boundary_order=1)
            der_term = der_term * prod ** term['pow'][j]
        if callable(term['coeff']) is True:
            der_term = term['coeff'](grid_points) * der_term
        else:
            der_term = term['coeff'] * der_term
        return der_term


class Derivative():
    def __init__(self, model):
        self.model = model

    def set_strategy(self, strategy):
        if strategy == 'NN':
            return Derivative_NN(self.model)

        elif strategy == 'autograd':
            return  Derivative_autograd(self.model)

        elif strategy == 'mat':
            return Derivative_mat(self.model)
