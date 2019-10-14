import numpy as np


def circular_permutations(n_points, symmetries=1, exclude_front=False):
    if (((n_points % symmetries > 0) & (not exclude_front)) |
            (((n_points - 1) % symmetries > 0) & exclude_front)):
        raise RuntimeError('symmetries should divide number of permuted points')

    indices = np.arange(n_points)

    if exclude_front:
        indices = np.tile(indices, (n_points - 1, 1))
        for i in range(1, n_points - 1):
            indices[i, 1:] = np.roll(indices[i - 1, 1:], -1)

    else:
        indices = np.tile(indices, (n_points, 1))
        for i in range(1, n_points):
            indices[i] = np.roll(indices[i - 1], -1)

    return indices


def traversal_permutations(graph, start_node, symmetry=1, max_depth=3):
    permutations = []
    adjacent = graph.adjacency()[start_node]

    if len(adjacent) % symmetry:
        raise RuntimeError()

    for i in range(len(adjacent) // symmetry):
        permutations.append(graph.traversals(max_depth=max_depth, start_nodes=start_node, roll_start=i)[0])

    return np.array(permutations)


class Template(object):

    def __init__(self, points, labels=None, permutations=None):
        self._points = points
        self._permutations = permutations
        self.labels = labels
        self._scale = None
        self._update_scale()

    def __len__(self):
        if self.permutations is None:
            return len(self.points)
        else:
            return self.permutations.shape[1]

    @property
    def points(self):
        return self._points

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        if labels is not None:
            labels = np.array(labels, dtype=np.int)
            if labels.shape != (len(self),):
                raise RuntimeError()

        self._labels = labels

    @property
    def permutations(self):
        return self._permutations

    @permutations.setter
    def permutations(self, permutations):
        if permutations is not None:
            permutations = np.array(permutations, dtype=np.int)
            if permutations.shape[1] <= len(self):
                raise RuntimeError()

        self._permutations = permutations

    @property
    def scale(self):
        return self._scale

    def generate_permutations(self):
        if self.permutations is None:
            yield self.points, self.labels

        else:
            if self.labels is None:
                for permutation in self.permutations:
                    yield self.points[permutation], None
            else:
                for permutation in self.permutations:
                    yield self.points[permutation], self.labels[permutation]

    def _update_scale(self):
        self._scale = np.sqrt(np.sum(np.linalg.norm(self._points, axis=1) ** 2))

    def add_points(self, points):
        if len(points.shape) == 1:
            points = points.reshape((1, 2))

        self._points = np.vstack((self._points, points))


def regular_polygon(sidelength, n, center=False):
    if center:
        points = np.zeros((n + 1, 2))
    else:
        points = np.zeros((n, 2))

    i = np.arange(n, dtype=np.int32)
    A = sidelength / (2 * np.sin(np.pi / n))

    if center:
        points[1:, 0] = A * np.cos(i * 2 * np.pi / n)
        points[1:, 1] = A * np.sin(i * 2 * np.pi / n)
    else:
        points[:, 0] = A * np.cos(i * 2 * np.pi / n)
        points[:, 1] = A * np.sin(i * 2 * np.pi / n)

    return Template(points)


def repeat_positions(positions, cell, n, m):
    N = len(positions)

    n0, n1 = 0, n
    m0, m1 = 0, m
    new_positions = np.zeros((n * m * len(positions), 2), dtype=np.float)

    new_positions[:N] = positions.copy()

    k = N
    for i in range(n0, n1):
        for j in range(m0, m1):
            if i + j != 0:
                l = k + N
                new_positions[k:l] = positions + np.dot((i, j), cell)
                k = l

    new_cell = np.array(cell).copy()
    new_cell[0] *= n
    new_cell[1] *= m

    return new_positions, new_cell


def wrap_positions(positions, cell, center=(0.5, 0.5), eps=1e-7):
    if not hasattr(center, '__len__'):
        center = (center,) * 2

    shift = np.asarray(center) - 0.5 - eps

    fractional = np.linalg.solve(cell.T, np.asarray(positions).T).T - shift

    for i in range(2):
        fractional[:, i] %= 1.0
        fractional[:, i] += shift[i]

    return np.dot(fractional, cell)
