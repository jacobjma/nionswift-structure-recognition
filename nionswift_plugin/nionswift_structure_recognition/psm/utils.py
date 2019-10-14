import numba as nb
import numpy as np


def generate_indices(labels):
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(0, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')

    for i, (l, h) in enumerate(zip(lo, hi)):
        yield i, indices[l:h]


@nb.njit
def set_difference(ar1, ar2):
    mask = np.full(len(ar1), True)
    for a in ar2:
        mask &= (ar1 != a)
    return ar1[mask]


@nb.njit
def check_clockwise(polygon):
    clockwise = False
    signed_area = 0.
    for i in range(len(polygon)):
        signed_area += polygon[i - 1, 0] * polygon[i, 1] - polygon[i, 0] * polygon[i - 1, 1]
    if signed_area > 0.:
        clockwise = True
    return clockwise


#@nb.njit
def simplex_circumcenter(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    a = np.linalg.det([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
    bx = -np.linalg.det([[x1 ** 2 + y1 ** 2, y1, 1],
                         [x2 ** 2 + y2 ** 2, y2, 1],
                         [x3 ** 2 + y3 ** 2, y3, 1]])

    by = np.linalg.det([[x1 ** 2 + y1 ** 2, x1, 1],
                        [x2 ** 2 + y2 ** 2, x2, 1],
                        [x3 ** 2 + y3 ** 2, x3, 1]])
    x0 = -bx / (2 * a)
    y0 = -by / (2 * a)
    return np.array([x0, y0])


@nb.njit
def ind2sub(n, ind):
    rows = (ind // n)
    cols = (ind % n)
    return (rows, cols)


@nb.njit
def sub2ind(n, rows, cols):
    return rows * n + cols
