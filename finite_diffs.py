from copy import copy
from typing import Union, List, Tuple

flatten_list = lambda t: [item for sublist in t for item in sublist]


class Finite_diffs():
    """
    Implements the Finite Difference method for a given operator. \n
    `finite_diff_shift`, `scheme_build`, `sign_order` implement 1st order (in terms of accuracy) finite difference. \n
    `second_order_shift`, `second_order_scheme_build`, `second_order_sign_order` implement respectively 2nd order (in terms of accuracy) finite difference.
    """
    # the idea is simple - central difference changes [0]->([1]-[-1])/(2h) (in terms of grid nodes position)
    @staticmethod
    def finite_diff_shift(diff: list, axis: int, mode: str) ->  list:
        """
        1st order points shift for the corresponding finite difference mode.

        Args:
            diff: values of finite differences.
            axis: axis.
            mode: the finite difference mode (i.e., forward, backward, central).

        Returns:
            diff_list: list with shifted points.
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

    @staticmethod
    def scheme_build(axes: list, varn: int, axes_mode: str) -> Tuple[list, list]:
        """
        Building first order (in terms of accuracy) finite-difference stencil.

        Args:
            axes: axes that transforms using FDM. (operator in conventional form)
            varn: Dimensionality of the problem.
            axes_mode: 'central' or combination of 'f' and 'b'.

        Returns:
            - finite_diff: transformed axes due to finite difference mode.
            - direction_list: list, which contains directions (i.e, 'central', 'f', 'b').
        """
        order = len(axes)
        finite_diff = []
        direction_list = []
        # we generate [0,0,0,...] for number of variables (varn)
        for i in range(varn):
            finite_diff += [0]
        # just to make this [[0,0,...]]
        finite_diff = [finite_diff]
        # when we increase differential order
        for i in range(order):
            diff_list = []
            for diff in finite_diff:
                # we use [0,0]->[[1,0],[-1,0]] rule for the axis
                if axes_mode == 'central':
                    f_diff = Finite_diffs.finite_diff_shift(diff, axes[i], 'central')
                else:
                    f_diff = Finite_diffs.finite_diff_shift(diff, axes[i], axes_mode[axes[i]])
                if len(diff_list) == 0:
                    # and put it to the pool of differentials if it is empty
                    diff_list = f_diff
                else:
                    # or add to the existing pool
                    for diffs in f_diff:
                        diff_list.append(diffs)
            # there we go to the next differential if needed
            finite_diff = diff_list
            direction_list.append(axes_mode[axes[i]])
        return finite_diff, direction_list

    @staticmethod
    def sign_order(order: list, mode: str, h: float = 1 / 2) -> list:
        """
        Determines the sign of the derivative for the corresponding transformation from Finite_diffs.scheme_build()

        Args:
            order: order of differentiation.
            mode: calculation type of finite difference.
            h: discretizing parameter in finite difference method (i.e., grid resolution for scheme).

        Returns:
            sign_list: list, with signs for corresponding points.

        """
        sign_list = [1]
        for i in range(order):
            start_list = []
            for sign in sign_list:
                if mode == 'central':
                    start_list.append([sign * (1 / (2 * h)), -sign * (1 / (2 * h))])
                else:
                    start_list.append([sign / h, -sign / h])
            sign_list = flatten_list(start_list)
        return sign_list

    @staticmethod
    def second_order_shift(diff: list, axis: int, mode: str) -> list:
        """
        2nd order points shift for the corresponding finite difference mode.

        Args:
            diff: values of finite differences.
            axis: axis.
            mode: the finite difference mode (i.e., forward, backward, central).

        Returns:
            diff_list: list with shifted points.
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

    @staticmethod
    def second_order_scheme_build(axes: list, varn: int, axes_mode: str) -> tuple[list, list]:
        """
        Building second order (in terms of accuracy) finite-difference stencil.

        Args:
            axes: axes that transforms using FDM. (operator in conventional form)
            varn: dimensionality of the problem.
            axes_mode: 'central' or combination of 'f' and 'b'.

        Returns:
            finite_diff: transformed axes due to finite difference mode.
            direction_list: list, which contains directions (i.e, 'central', 'f', 'b').
        """
        order = len(axes)
        finite_diff = []
        direction_list = []
        # we generate [0,0,0,...] for number of variables (varn)
        for i in range(varn):
            finite_diff += [0]
        # just to make this [[0,0,...]]
        finite_diff = [finite_diff]
        # when we increase differential order
        for i in range(order):
            diff_list = []
            for diff in finite_diff:
                # we use [0,0]->[[1,0],[-1,0]] rule for the axis
                if axes_mode == 'central':
                    f_diff = Finite_diffs.second_order_shift(diff, axes[i], 'central')
                else:
                    f_diff = Finite_diffs.second_order_shift(diff, axes[i], axes_mode[axes[i]])
                if len(diff_list) == 0:
                    # and put it to the pool of differentials if it is empty
                    diff_list = f_diff
                else:
                    # or add to the existing pool
                    for diffs in f_diff:
                        diff_list.append(diffs)
            # the we go to the next differential if needed
            finite_diff = diff_list
            direction_list.append(axes_mode[axes[i]])
        return finite_diff, direction_list

    @staticmethod
    def second_order_sign_order(order: list, mode: str, h: float = 1/2) -> list:
        """
        Determines the sign of the derivative for the corresponding point transformation from `Finite_diffs.scheme_build`.\n
        Same as `sign_order`, but more precise due to second order of accuracy.

        Args:
            order: order of differentiation.
            mode: calculation type of finite difference.
            h: discretizing parameter in finite difference method (i.e., grid resolution for scheme).

        Returns:
            sign_list: list, with signs for corresponding points.
        """
        sign_list = [1]
        for i in range(order):
            start_list = []
            for sign in sign_list:
                if mode[i] == 'f':
                    start_list.append([3 * (1 / (2 * h)) * sign, -4 * (1 / (2 * h)) * sign, (1 / (2 * h)) * sign])
                elif mode[i] == 'b':
                    start_list.append([-3 * (1 / (2 * h)) * sign, 4 * (1 / (2 * h)) * sign, -(1 / (2 * h)) * sign])
            sign_list = flatten_list(start_list)
        return sign_list
