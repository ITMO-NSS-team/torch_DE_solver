""" Module for construct a domain with complex or irregular geometry"""

import torch


def is_inside_circle(coord, circ_geom):
    """
    Checks if a point is inside a circle.

    Args:
        coord (torch.Tensor): coordinates of the point.
        circ_geom (list or tuple): geometry of the circle,
            specified as a list/tuple [center_x, center_y, ..., radius].

    Returns:
        bool: True if the point is inside the circle, False otherwise.
    """

    center = torch.tensor(circ_geom[0: -1], dtype=torch.float32)
    radius = torch.tensor(circ_geom[-1], dtype=torch.float32)

    num_dim = center.shape[0]

    return torch.sum(torch.tensor([(coord[i] - center[i]) ** 2 for i in range(num_dim)])) < radius ** 2


def is_on_circle_boundary(coord, circ_geom):
    """
    Checks if a point lies on the boundary of a circle.

    Args:
        coord (torch.Tensor): coordinates of the point.
        circ_geom (list or tuple): geometry of the circle,
            specified as a list/tuple [center_x, center_y, ..., radius].

    Returns:
        bool: True if the point is on the circle boundary, False otherwise.
    """

    center = torch.tensor(circ_geom[0], dtype=torch.float32)
    radius = torch.tensor(circ_geom[-1], dtype=torch.float32)

    num_dim = center.shape[0]

    return torch.isclose(
        torch.sum(torch.tensor([(coord[i] - center[i]) ** 2 for i in range(num_dim)])),
        radius ** 2,
        rtol=1e-4
    )


def is_inside_rectangle(coord, rect_geom):
    """
    Checks if a point is inside a rectangle.

    Args:
        coord (torch.Tensor): coordinates of the point.
        rect_geom (list or tuple): geometry of the rectangle,
            specified as [lower_corner, upper_corner], where
            lower_corner and upper_corner are lists/tuples of coordinates.

    Returns:
        bool: True if the point is inside the rectangle, False otherwise.
    """

    in_rec = []
    for i in range(len(coord)):
        in_rec.append(rect_geom[0][i] <= coord[i] <= rect_geom[1][i])

    return all(in_rec)


def csg_domain_difference(grid, geom_figure):
    """
    Constructs a grid by excluding a specified geometric region.

    Args:
        grid (torch.Tensor): coordinates of the original grid as a list of points.
        geom_figure (dict): geometry of the figure to exclude.
            The dictionary should include:
                - 'name': type of the figure.
                - 'coords': coordinates defining the geometry.

    Returns:
        torch.Tensor: grid excluding the specified geometric region.
    """

    csg_grid = []

    for point in grid:
        if geom_figure['name'] == 'rectangle':
            if not is_inside_rectangle(point, geom_figure['coords']):
                csg_grid.append([float(p) for p in point])
        elif geom_figure['name'] == 'circle':
            if not is_inside_circle(point, geom_figure['coords']):
                csg_grid.append([float(p) for p in point])

    csg_grid = torch.tensor(csg_grid, dtype=torch.float32)

    return csg_grid


def csg_boundary_circle(grid, geom_figure):
    """
    Extracts points that lie on the boundary of a circle.

    Args:
        grid (torch.Tensor): coordinates of the original grid as a list of points.
        geom_figure (dict): geometry of the circle.
            The dictionary should include:
                - 'center': center of the circle as a list/tuple of coordinates.
                - 'radius': radius of the circle.

    Returns:
        torch.Tensor: points that lie on the circle boundary.
    """

    csg_bnd = []

    center = geom_figure['center']
    radius = geom_figure['radius']

    for point in grid:
        if is_on_circle_boundary(point, (center, radius)):
            csg_bnd.append([float(p) for p in point])

    return torch.tensor(csg_bnd, dtype=torch.float32)


