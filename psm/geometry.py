import numba as nb
import numpy as np


def polygon_area(points):
    return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))


def points_in_bounding_box(points, bounding_box):
    bounding_box = np.array(bounding_box)
    return ((points[:, 0] > bounding_box[0, 0]) & (points[:, 1] > bounding_box[1, 0]) &
            (points[:, 0] < bounding_box[0, 1]) & (points[:, 1] < bounding_box[1, 1]))


@nb.jit(nopython=True)
def bounding_box_from_points(points, margin=0):
    return np.array([[np.min(points[:, 0]) - margin, np.max(points[:, 0]) + margin],
                     [np.min(points[:, 1]) - margin, np.max(points[:, 1]) + margin]])


@nb.jit(nopython=True)
def point_in_bounding_box(point, bounding_box):
    if point[0] < bounding_box[0, 0]:
        return False
    elif point[0] > bounding_box[0, 1]:
        return False
    elif point[1] < bounding_box[1, 0]:
        return False
    elif point[1] > bounding_box[1, 1]:
        return False
    return True


@nb.jit(nopython=True)
def any_point_in_polygon(points, polygon):
    bounding_box = bounding_box_from_points(polygon)
    for i in range(len(points)):
        if point_in_bounding_box(points[i], bounding_box):
            if point_in_polygon(points[i], polygon):
                return True
    return False


@nb.njit
def check_clockwise(polygon):
    clockwise = False
    signed_area = 0.
    for i in range(len(polygon)):
        signed_area += polygon[i - 1, 0] * polygon[i, 1] - polygon[i, 0] * polygon[i - 1, 1]
    if signed_area > 0.:
        clockwise = True
    return clockwise


@nb.jit(nopython=True)
def point_in_polygon(point, polygon):
    n = len(polygon)
    inside = False
    xints = 0.0
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if point[1] > min(p1y, p2y):
            if point[1] <= max(p1y, p2y):
                if point[0] <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or point[0] <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def points_in_polygon(points, polygon, return_indices=False):
    if return_indices:
        return np.array([i for i, point in enumerate(points) if point_in_polygon(point, polygon)])
    else:
        return np.array([point for point in points if point_in_polygon(point, polygon)])


def regular_polygon(sidelength, n):
    points = np.zeros((n, 2))

    i = np.arange(n, dtype=np.int32)
    A = sidelength / (2 * np.sin(np.pi / n))

    points[:, 0] = A * np.cos(i * 2 * np.pi / n)
    points[:, 1] = A * np.sin(i * 2 * np.pi / n)

    return points


def kabsch_orientation(src, dst):
    A = np.dot(dst.T, src)

    V, S, W = np.linalg.svd(A)

    if (np.linalg.det(V) * np.linalg.det(W)) < 0.:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    U = np.dot(V, W)
    return np.arctan2(-U[0, 1], U[0, 0])
