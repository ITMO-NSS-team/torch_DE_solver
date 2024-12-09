""" Module for construct a domain with complex geometry"""

import torch


def inside_circle(coord, circ_geom):
    center = torch.tensor(circ_geom[0: -1], dtype=torch.float32)
    radius = torch.tensor(circ_geom[-1], dtype=torch.float32)

    num_dim = center.shape[0]

    return torch.sum(torch.tensor([(coord[i] - center[i]) ** 2 for i in range(num_dim)])) < radius ** 2


def boundary_circle(coord, circ_geom):
    center = torch.tensor(circ_geom[0], dtype=torch.float32)
    radius = torch.tensor(circ_geom[-1], dtype=torch.float32)

    num_dim = center.shape[0]

    return torch.isclose(
        torch.sum(torch.tensor([(coord[i] - center[i]) ** 2 for i in range(num_dim)])),
        radius ** 2,
        rtol=1e-4
    )


def inside_rectangle(coord, rect_geom):
    in_rec = []
    for i in range(len(coord)):
        in_rec.append(rect_geom[0][i] <= coord[i] <= rect_geom[1][i])

    return all(in_rec)


def csg_domain_difference(grid, geom_figure):
    csg_grid = []

    for point in grid:
        if geom_figure['name'] == 'rectangle':
            if not inside_rectangle(point, geom_figure['coords']):
                csg_grid.append([float(p) for p in point])
        elif geom_figure['name'] == 'circle':
            if not inside_circle(point, geom_figure['coords']):
                csg_grid.append([float(p) for p in point])

    csg_grid = torch.tensor(csg_grid, dtype=torch.float32)

    return csg_grid


def csg_boundary_circle(grid, geom_figure):
    csg_bnd = []

    center = geom_figure['center']
    radius = geom_figure['radius']

    for point in grid:
        if boundary_circle(point, (center, radius)):
            csg_bnd.append([float(p) for p in point])

    return torch.tensor(csg_bnd, dtype=torch.float32)








