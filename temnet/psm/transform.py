import numpy as np


def rotate_points(points, angle, center='origin'):
    if isinstance(center, str):
        if center.lower() == 'cop':
            center = points.mean(axis=0)
        elif center.lower() == 'origin':
            center = np.array((0., 0.))
    else:
        center = np.array(center)

    cosa = np.cos(angle)
    sina = np.sin(angle)

    R = np.array([[cosa, sina], [-sina, cosa]])

    return np.dot(points - center, R.T) + center


def affine_transform(src, dst):
    coeffs = range(6)

    xs = src[:, 0]
    ys = src[:, 1]
    xd = dst[:, 0]
    yd = dst[:, 1]
    rows = src.shape[0]

    # params: a0, a1, a2, b0, b1, b2, c0, c1
    A = np.zeros((rows * 2, 9))
    A[:rows, 0] = xs
    A[:rows, 1] = ys
    A[:rows, 2] = 1
    A[:rows, 6] = - xd * xs
    A[:rows, 7] = - xd * ys
    A[rows:, 3] = xs
    A[rows:, 4] = ys
    A[rows:, 5] = 1
    A[rows:, 6] = - yd * xs
    A[rows:, 7] = - yd * ys
    A[:rows, 8] = xd
    A[rows:, 8] = yd

    A = A[:, list(coeffs) + [8]]

    _, _, V = np.linalg.svd(A)

    H = np.zeros((3, 3))
    H.flat[list(coeffs) + [8]] = - V[-1, :-1] / V[-1, -1]
    H[2, 2] = 1
    return H
