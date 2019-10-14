import numpy as np


def repeat_positions(positions, cell, n, m):
    N = len(positions)
    new_positions = np.zeros((n * m * len(positions), 2), dtype=np.float)
    new_positions[:N] = positions.copy()

    k = N
    for i in range(0, n):
        for j in range(0, m):
            if i + j != 0:
                l = k + N
                new_positions[k:l] = positions + np.dot((i, j), cell)
                k = l

    new_cell = cell.copy()
    new_cell[0] *= n
    new_cell[1] *= m

    return new_positions, new_cell


def wrap_positions(positions, cell, eps=1e-7):
    fractional = np.linalg.solve(cell.T, np.asarray(positions).T).T - eps

    for i in range(2):
        fractional[:, i] %= 1.0
        fractional[:, i] += eps

    return np.dot(fractional, cell)


def orthogonal_graphene(a=2.46):
    basis = [(0, 0), (2 / 3., 1 / 3.)]
    cell = np.array([[a, 0], [-a / 2., a * 3 ** 0.5 / 2.]])

    positions = np.dot(np.array(basis), np.array(cell))
    positions, cell = repeat_positions(positions, cell, 2, 2)

    cell[1, 0] = 0.
    positions = wrap_positions(positions, cell)

    return positions, cell


def is_position_inside(positions, cell):
    fractional = np.linalg.solve(cell.T, np.asarray(positions).T)
    return (fractional[0] > 0) & (fractional[1] > 0) & (fractional[0] < 1) & (fractional[1] < 1)


def rotate_positions(positions, cell, angle, center=None):
    if center is None:
        center = np.sum(cell, axis=1) / 2
    angle = angle / 180. * np.pi
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.dot(R, positions.T - np.array(center)[:, None]).T + center


def graphene_in_a_box(box, rotation=0., a=2.46, margin=0.):
    positions, cell = orthogonal_graphene(a=a)

    diagonal = np.hypot(box[0], box[1])
    n = 2 * np.ceil(diagonal / cell[0, 0]).astype(int)
    m = 2 * np.ceil(diagonal / cell[1, 1]).astype(int)

    positions, cell = repeat_positions(positions, cell, n, m)
    positions = rotate_positions(positions, cell, rotation)
    positions -= np.diag(cell) / 2

    cell[0, 0] = box[0] + 2 * margin
    cell[1, 1] = box[1] + 2 * margin
    positions = positions[is_position_inside(positions + margin, cell)]

    return positions, cell


def gaussian(x, b):
    return np.exp(-x / b)


def gaussian_superposition(positions, shape, origin, extent, width):
    shape = np.array(shape)
    origin = np.array(origin)
    extent = np.array(extent)

    positions = positions - origin

    sampling = extent / shape

    margin = np.int32(np.ceil(3 * width / sampling.min()))
    width = 2 * width ** 2

    y = np.linspace(0, extent[0] + 2 * margin * sampling[0] - sampling[0], shape[0] + 2 * margin)
    x = np.linspace(0, extent[1] + 2 * margin * sampling[1] - sampling[1], shape[1] + 2 * margin)

    image = np.zeros((shape[0] + 2 * margin, shape[1] + 2 * margin))

    pixel_positions = positions / sampling

    for i in range(len(positions)):
        position = positions[i]
        pixel_position = pixel_positions[i]

        y_lim_min = np.int32(np.floor(pixel_position[0]))
        y_lim_max = np.int32(np.floor(pixel_position[0] + 2 * margin + 1))
        x_lim_min = np.int32(np.floor(pixel_position[1]))
        x_lim_max = np.int32(np.floor(pixel_position[1] + 2 * margin + 1))

        xi = x[x_lim_min:x_lim_max]
        yi = y[y_lim_min:y_lim_max]

        distances = ((position[0] + margin * sampling[0] - yi).reshape(1, -1) ** 2 +
                     (position[1] + margin * sampling[1] - xi).reshape(-1, 1) ** 2)

        image[y_lim_min: y_lim_max, x_lim_min: x_lim_max] += gaussian(distances, width)

    return image[margin:-margin, margin:-margin]
