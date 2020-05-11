import numba as nb
import numpy as np


@nb.njit
def rmsd_qcp(src, dst):
    M = np.dot(dst.T, src)

    xx, xy = M[0, :]
    yx, yy = M[1, :]

    xx_yy = xx + yy
    xy_yx = xy - yx
    xy_yx_2 = xy_yx ** 2

    xx_yy_u = xx_yy + np.sqrt(xy_yx_2 + xx_yy ** 2)
    xx_yy_u_2 = xx_yy_u ** 2

    denom = xx_yy_u_2 + xy_yx_2

    Uxx = (xx_yy_u_2 - xy_yx_2) / denom
    Uxy = 2 * xy_yx * xx_yy_u / denom

    U = np.array([[Uxx, -Uxy], [Uxy, Uxx]])

    rmsd = np.sqrt(np.sum((np.dot(src, U) - dst) ** 2) / len(src))
    return rmsd


def kabsch_orientation(src, dst):
    A = np.dot(dst.T, src)

    V, S, W = np.linalg.svd(A)

    if (np.linalg.det(V) * np.linalg.det(W)) < 0.:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    U = np.dot(V, W)
    return np.arctan2(-U[0, 1], U[0, 0])


def rotate(points, angle):
    center = points.mean(axis=0)
    R = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.dot(R, points.T - np.array(center)[:, None]).T + center


def pairwise_rmsd(A, B, transform='rigid', scale_invariant=True, pivot='cop'):
    rmsd = np.zeros((len(A), len(B)))
    rmsd[:] = np.inf

    if pivot == 'cop':
        A = [a - np.mean(a, axis=0) for a in A]
        B = [b - np.mean(b, axis=0) for b in B]

    elif pivot == 'front':
        A = [a - a[0] for a in A]
        B = [b - b[0] for b in B]

    else:
        raise RuntimeError('pivot must be "cop" or "front"')

    A_scales = [np.sqrt(np.sum(np.linalg.norm(a, axis=1) ** 2)) for a in A]
    B_scales = [np.sqrt(np.sum(np.linalg.norm(b, axis=1) ** 2)) for b in B]

    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if len(a) == len(b):
                if (transform == 'rigid') & scale_invariant:
                    scale = np.sqrt(A_scales[i] ** 2 + B_scales[j] ** 2)
                    a_ = a / scale
                    b_ = b / scale

                elif (transform == 'similarity') & scale_invariant:
                    a_ = a / A_scales[i]
                    b_ = b / B_scales[j]

                elif (transform == 'similarity') & (not scale_invariant):
                    a_ = a.copy()
                    b_ = b * A_scales[i] / B_scales[j]

                else:
                    raise RuntimeError('transform must be "rigid" or "similarity"')

                rmsd[i, j] = rmsd_qcp(a_, b_)

    return rmsd


def regular_polygon(sidelength, n):
    points = np.zeros((n, 2))

    i = np.arange(n, dtype=np.int32)
    A = sidelength / (2 * np.sin(np.pi / n))

    points[:, 0] = A * np.cos(i * 2 * np.pi / n)
    points[:, 1] = A * np.sin(i * 2 * np.pi / n)
    return points


def order_adjacency_clockwise(points, adjacency, counter_clockwise=False):
    for node, adjacent in enumerate(adjacency):
        centered = points[adjacent] - points[node]
        order = np.arctan2(centered[:, 0], centered[:, 1])
        adjacency[node] = [x for _, x in sorted(zip(order, adjacent), reverse=counter_clockwise)]
    return adjacency


def polygon_area(points):
    return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))
