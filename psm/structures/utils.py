import numpy as np


def repeat(points, cell, n, m):
    N = len(points)

    n0, n1 = 0, n
    m0, m1 = 0, m
    new_points = np.zeros((n * m * N, 2), dtype=np.float)
    new_points[:N] = points

    k = N
    for i in range(n0, n1):
        for j in range(m0, m1):
            if i + j != 0:
                l = k + N
                new_points[k:l] = points + np.dot(np.array((i, j)), cell)
                k = l

    cell = cell * np.array((n, m))

    return new_points, cell


def wrap(points, cell, center=(0.5, 0.5), eps=1e-7):
    if not hasattr(center, '__len__'):
        center = (center,) * 2

    shift = np.asarray(center) - 0.5 - eps

    fractional = np.linalg.solve(cell.T, np.asarray(points).T).T - shift

    for i in range(2):
        fractional[:, i] %= 1.0
        fractional[:, i] += shift[i]

    points = np.dot(fractional, cell)
    return points


def rotate(points, angle, cell=None, center=None, rotate_cell=False):
    if center is None:
        if cell is None:
            center = np.array([0., 0.])
        else:
            center = cell.sum(axis=1) / 2

    angle = angle / 180. * np.pi
    R = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = np.dot(R, points.T - np.array(center)[:, None]).T + center
    if rotate_cell:
        cell = np.dot(R, cell.T).T

    if cell is None:
        return points
    else:
        return points, cell


def fill_rectangle(points, cell, extent, origin=None, margin=0., eps=1e-17):
    if origin is None:
        origin = np.zeros(2)

    P_inv = np.linalg.inv(cell)

    origin_t = np.dot(origin, P_inv)
    origin_t = origin_t % 1.0

    lower_corner = np.dot(origin_t, cell)
    upper_corner = lower_corner + extent

    corners = np.array([[-margin - eps, -margin - eps],
                        [upper_corner[0].item() + margin + eps, -margin - eps],
                        [upper_corner[0].item() + margin + eps, upper_corner[1].item() + margin + eps],
                        [-margin - eps, upper_corner[1].item() + margin + eps]])
    n0, m0 = 0, 0
    n1, m1 = 0, 0
    for corner in corners:
        new_n, new_m = np.ceil(np.dot(corner, P_inv)).astype(np.int)
        n0 = max(n0, new_n)
        m0 = max(m0, new_m)
        new_n, new_m = np.floor(np.dot(corner, P_inv)).astype(np.int)
        n1 = min(n1, new_n)
        m1 = min(m1, new_m)

    points, _ = repeat(points, cell, (1 + n0 - n1).item(), (1 + m0 - m1).item())

    points = points + cell[0] * n1 + cell[1] * m1

    inside = ((points[:, 0] > lower_corner[0] + eps - margin) &
              (points[:, 1] > lower_corner[1] + eps - margin) &
              (points[:, 0] < upper_corner[0] + margin) &
              (points[:, 1] < upper_corner[1] + margin))
    new_points = points[inside] - lower_corner

    cell = np.array([[extent[0], 0], [0, extent[1]]])

    return new_points, cell
