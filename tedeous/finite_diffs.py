"""Module for subgrid creation corresponding to numerical scheme. It's used only *NN* method."""

from copy import  copy
import numpy as np

flatten_list = lambda t: [item for sublist in t for item in sublist]


class First_order_scheme():
    """
    Class for constructing a numerical scheme for solving differential equations. It employs a central difference scheme for interior points and forward or backward difference schemes for boundary points, depending on the specified point type.
    """


    def __init__(self, term: list, nvars: int, axes_scheme_type: str):
        """
        Initializes the first-order finite difference scheme.
        
                This scheme approximates the derivatives in a differential equation using finite differences.
                The initialization sets up the differentiation directions and the scheme type for each axis.
                This setup is crucial for accurately representing the differential equation within the neural network solver.
        
                Args:
                    term (list): Differentiation direction for each variable. For example, [0, 0] represents d^2u/dx^2 if x is the first direction in the grid.
                    nvars (int): The number of independent variables in the differential equation. For example, if the grid is (x, t), then nvars = 2.
                    axes_scheme_type (str): The type of finite difference scheme to use along each axis. Can be 'central' for a central difference scheme, or a combination of 'f' (forward) and 'b' (backward) for each axis.
        
                Returns:
                    None
        """

        self.term = term
        self.nvars = nvars
        if axes_scheme_type == 'central':
            self.direction_list = ['central' for _ in self.term]
        else:
            self.direction_list = [axes_scheme_type[i] for i in self.term]

    # the idea is simple - central difference changes
    # [0]->([1]-[-1])/(2h) (in terms of grid nodes position)
    @staticmethod
    def _finite_diff_shift(diff: list, axis: int, mode: str) ->  list:
        """
        Adjusts a point in the computational grid based on the finite difference scheme.
        
        This function shifts a specified coordinate within a grid point representation
        to calculate derivatives using finite difference approximations. The shift
        direction depends on the chosen finite difference mode (forward, backward, or central).
        This is essential for accurately estimating derivatives at different points in the domain
        when solving differential equations.
        
        Args:
            diff (list): A list representing the coordinates of a point in the grid.
            axis (int): The axis along which the shift is applied.
            mode (str): The finite difference mode ('forward', 'backward', or 'central').
        
        Returns:
            list: A list containing two shifted point representations. The first element
                corresponds to the point shifted in the positive direction (diff_p), and
                the second element corresponds to the point shifted in the negative
                direction (diff_m). If the mode is 'forward' only diff_p is shifted.
                If the mode is 'backward' only diff_m is shifted.
        """
 
        diff_p = copy(diff)
        diff_m = copy(diff)
        if mode == 'central':
            diff_p[axis] = diff_p[axis] + 1
            diff_m[axis] = diff_m[axis] - 1
        elif mode == 'f':
            diff_p[axis] = diff_p[axis] + 1
        elif mode == 'b':
            diff_m[axis] = diff_m[axis] - 1
        return [diff_p, diff_m]

    def scheme_build(self) -> list:
        """
        Builds a finite-difference scheme to represent derivatives for the neural network-based differential equation solver.
        
                This method constructs a numerical scheme that approximates the derivatives in the differential equation.
                It starts with a list of zeros, where the number of zeros corresponds to the number of variables.
                Then, it modifies the values along the axes corresponding to each term in the differential equation.
                For example, [0,0] -> [[1,0], [-1,0]] indicates that the term was [0] (d/dx) and the scheme type is 'central'.
                This scheme is used to compute the derivatives required by the neural network during the solution process.
        
                Args:
                    self (First_order_scheme): Instance of the First_order_scheme class, containing information about the differential equation and desired scheme.
        
                Returns:
                    list: A list of lists representing the numerical scheme. Each inner list corresponds to a finite difference approximation of a derivative.
        """

        order = len(self.term)
        finite_diff = [[0 for _ in range(self.nvars)]]
        for i in range(order):
            diff_list = []
            for diff in finite_diff:
                # we use [0,0]->[[1,0],[-1,0]] rule for the axis
                f_diff = self._finite_diff_shift(
                    diff, self.term[i], self.direction_list[i])

                if len(diff_list) == 0:
                    # and put it to the pool of differentials if it is empty
                    diff_list = f_diff
                else:
                    # or add to the existing pool
                    for diffs in f_diff:
                        diff_list.append(diffs)
            # then we go to the next differential if needed
            finite_diff = diff_list
        return finite_diff

    def sign_order(self, h: float = 1 / 2) -> list :
        """
        Determines the sign of the derivative for each term in the finite difference scheme,
        which is crucial for constructing the overall approximation of the differential equation's solution.
        
        The method starts with a positive sign (+1) and iteratively refines it based on the order of the differential.
        Each +1 transforms into [+1, -1] when the order increases, and each -1 transforms into [-1, +1].
        This process ensures that the signs alternate correctly to capture the derivative's behavior.
        
        Args:
            h (float, optional): Discretization parameter used in the finite difference method. Defaults to 1/2.
        
        Returns:
            list: A list containing the signs (+1 or -1) corresponding to each point in the finite difference scheme.
                  These signs are essential for correctly weighting the contributions of each point when approximating the derivative.
        """

        sign_list = [1]
        for _ in range(len(self.term)):
            start_list = []
            for sign in sign_list:
                if np.unique(self.direction_list)[0] == 'central':
                    start_list.append([sign * (1 / (2 * h)),
                                       -sign * (1 / (2 * h))])
                else:
                    start_list.append([sign / h, -sign / h])
            sign_list = flatten_list(start_list)
        return sign_list


class Second_order_scheme():
    """
    Crank–Nicolson method. This realization only for boundary points.
    """

    def __init__(self, term: list, nvars: int, axes_scheme_type: str):
        """
        Initializes a second-order finite difference scheme for approximating derivatives in a neural network-based differential equation solver.
        
                This scheme is used to calculate the derivatives required for the loss function, enabling the neural network to learn the solution of the differential equation.
        
                Args:
                    term (list): Specifies the differentiation direction as a list of integers. For example, [0, 0] corresponds to d²u/dx², assuming 'x' is the first direction in the grid.
                    nvars (int): Indicates the number of independent variables in the problem. For instance, if the grid is defined by grid(x, t), then nvars = 2.
                    axes_scheme_type (str): Defines the type of finite difference scheme to use along each axis. It can be 'central' for a central difference scheme or a combination of 'f' (forward) and 'b' (backward) for one-sided schemes.
        
                Raises:
                    ValueError: If an unsupported scheme type is provided.
        """
        
        self.term = term
        self.nvars = nvars
        try:
            axes_scheme_type == 'central'
        except:
            print('These scheme only for "f" and "b" points')
            raise ValueError
        self.direction_list = [axes_scheme_type[i] for i in self.term]

    @staticmethod
    def _second_order_shift(diff, axis, mode) -> list:
        """
        Shifts points based on the finite difference mode for second-order schemes. This adjustment is crucial for accurately calculating derivatives using finite difference approximations within the neural network-based differential equation solver. By shifting the points, we ensure that the finite difference calculations align correctly with the network's internal representation, leading to more precise solutions.
        
                Args:
                    diff (list): Values of finite differences.
                    axis (int): Axis along which the shift is applied.
                    mode (str): Finite difference mode ('f' for forward, 'b' for backward).
        
                Returns:
                    list: A list containing three shifted point sets, corresponding to the second-order finite difference scheme.
        """
        diff_1 = copy(diff)
        diff_2 = copy(diff)
        diff_3 = copy(diff)
        if mode == 'f':
            diff_3[axis] = diff_3[axis] + 2
            diff_2[axis] = diff_2[axis] + 1
        elif mode == 'b':
            diff_3[axis] = diff_3[axis] - 2
            diff_2[axis] = diff_2[axis] - 1
        else:
            print('Wrong mode')
        return [diff_3, diff_2, diff_1]

    def scheme_build(self) -> list:
        """
        Builds the numerical scheme for the Crank-Nicolson variant, crucial for discretizing the differential equation within the neural network solver. This scheme mirrors the first-order method's approach but incorporates a shift determined by 'second_order_shift' to enhance accuracy.
        
                Args:
                    self (Second_order_scheme): An instance of the Second_order_scheme class containing the terms, directions, and number of variables for building the scheme.
        
                Returns:
                    list: A list of numerical schemes, each representing a discrete approximation of the differential equation, ready for use in the neural network solver.
        """

        order = len(self.term)
        finite_diff = [[0 for _ in range(self.nvars)]]
        # when we increase differential order
        for i in range(order):
            diff_list = []
            for diff in finite_diff:
                # we use [0,0]->[[1,0],[-1,0]] rule for the axis
                f_diff = self._second_order_shift(
                    diff, self.term[i], self.direction_list[i])
                if len(diff_list) == 0:
                    # and put it to the pool of differentials if it is empty
                    diff_list = f_diff
                else:
                    # or add to the existing pool
                    for diffs in f_diff:
                        diff_list.append(diffs)
            # then we go to the next differential if needed
            finite_diff = diff_list
        return finite_diff

    def sign_order(self, h: float = 1/2) -> list:
        """
        Generates the coefficients for a second-order finite difference scheme used to approximate derivatives within a neural network-based differential equation solver.
        
        This method constructs the coefficients based on the specified forward or backward differences and discretization parameter. These coefficients are essential for accurately representing the derivatives in the neural network's loss function, enabling it to learn the solution to the differential equation.
        
        Args:
            h (float, optional): The step size or grid resolution used in the finite difference approximation. Smaller values generally lead to more accurate approximations but may increase computational cost. Defaults to 1/2.
        
        Returns:
            list: A list of coefficients representing the finite difference approximation of the derivative at different points. These coefficients are used in the loss function to enforce the differential equation constraint.
        """

        sign_list = [1]
        for i in range(len(self.term)):
            start_list = []
            for sign in sign_list:
                if self.direction_list[i] == 'f':
                    start_list.append([3 * (1 / (2 * h)) * sign,
                                       -4 * (1 / (2 * h)) * sign,
                                       (1 / (2 * h)) * sign])
                elif self.direction_list[i] == 'b':
                    start_list.append([-3 * (1 / (2 * h)) * sign,
                                       4 * (1 / (2 * h)) * sign,
                                       -(1 / (2 * h)) * sign])
            sign_list = flatten_list(start_list)
        return sign_list


class Finite_diffs():
    """
    Class for numerical scheme choosing.
    """


    def __init__(self, term: list, nvars: int, axes_scheme_type: str):
        """
        Initializes a finite difference scheme for approximating derivatives in a neural network-based differential equation solver. This setup is crucial for translating derivative information into a format usable by the neural network.
        
                Args:
                    term (list): Specifies the order and direction of differentiation. For instance, [0,0] represents d2u/dx2 if x is the first variable in the grid.
                    nvars (int):  Indicates the number of independent variables in the differential equation. For example, if the grid is defined by grid(x,t), then nvars = 2.
                    axes_scheme_type (str):  Defines the finite difference scheme to be used along each axis ('central', 'forward' ('f'), or 'backward' ('b')).
        
                Returns:
                    None
        """

        self.term = term
        self.nvars = nvars
        self.axes_scheme_type = axes_scheme_type

    def scheme_choose(self, scheme_label: str, h:float = 1 / 2) -> list:
        """
        Selects a numerical scheme based on the specified order to approximate solutions of differential equations using neural networks.
        
        This method chooses between first-order and second-order finite difference schemes.
        The selection influences how the solution space is discretized and, consequently,
        how the neural network learns to approximate the solution.
        
        Args:
            scheme_label (str): '2' for the second-order scheme (applied at boundary points),
                '1' for the first-order scheme.
            h (float, optional): Discretization parameter (grid resolution). Defaults to 1/2.
        
        Returns:
            list: A list containing the numerical scheme and associated sign information.
                  The scheme is used to discretize the differential equation,
                  and the sign is relevant for the numerical stability and accuracy of the solution.
        """

        if self.term == [None]:
            return [[None], [1]]
        elif scheme_label == '2':
            cl_scheme = Second_order_scheme(self.term, self.nvars,
                                                        self.axes_scheme_type)
        elif scheme_label == '1':
            cl_scheme = First_order_scheme(self.term, self.nvars,
                                                        self.axes_scheme_type)

        scheme = cl_scheme.scheme_build()
        sign = cl_scheme.sign_order(h=h)
        return [scheme, sign]
