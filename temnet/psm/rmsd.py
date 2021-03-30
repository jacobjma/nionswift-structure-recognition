import numba as nb
import numpy as np


def rmsd_kabsch(p, q):
    A = np.dot(q.T, p)

    V, S, W = np.linalg.svd(A)

    if (np.linalg.det(V) * np.linalg.det(W)) < 0.:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    U = np.dot(V, W)

    rotated = np.dot(p, U.T)

    return np.sqrt(np.sum((rotated - q) ** 2) / len(p))


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


def batch_rmsd_qcp(src, dst):
    N = max(src.shape[0], dst.shape[0])

    M = np.matmul(np.swapaxes(dst, 1, 2), src)

    xx = M[:, 0, 0]
    xy = M[:, 0, 1]
    yx = M[:, 1, 0]
    yy = M[:, 1, 1]

    xx_yy = xx + yy
    xy_yx = xy - yx
    xy_yx_2 = xy_yx ** 2

    xx_yy_u = xx_yy + np.sqrt(xy_yx_2 + xx_yy ** 2)
    xx_yy_u_2 = xx_yy_u ** 2

    denom = xx_yy_u_2 + xy_yx_2

    Uxx = (xx_yy_u_2 - xy_yx_2) / denom
    Uxy = 2 * xy_yx * xx_yy_u / denom

    U = np.zeros((N, 2, 2))
    U[:, 0, 0] = Uxx
    U[:, 1, 1] = Uxx
    U[:, 0, 1] = -Uxy
    U[:, 1, 0] = Uxy

    rmsd = np.sqrt(np.sum((np.matmul(src, U) - dst) ** 2, axis=(1, 2)) / src.shape[1])
    return rmsd


def pairwise_rmsd(P, Q, P_labels=None, Q_labels=None, transform='rigid', scale_invariant=True, pivot='cop'):
    if P_labels is not None:
        if len(P_labels) != len(P):
            raise RuntimeError()
        P_labels = [np.array(a_labels) for a_labels in P_labels]

    rmsd = np.zeros((len(P), len(Q)))
    rmsd[:] = np.inf

    if pivot == 'cop':
        P = [p - np.mean(p, axis=0) for p in P]
        Q = [q - np.mean(q, axis=0) for q in Q]

    elif pivot == 'front':
        P = [p - p[0] for p in P]
        Q = [q - q[0] for q in Q]

    else:
        raise RuntimeError('pivot must be "cop" or "front"')

    P_scales = [np.sqrt(np.sum(np.linalg.norm(p, axis=1) ** 2)) for p in P]
    Q_scales = [np.sqrt(np.sum(np.linalg.norm(q, axis=1) ** 2)) for q in Q]

    for i, p in enumerate(P):
        for j, q in enumerate(Q):
            if len(p) != len(q):
                continue

            if (P_labels is not None) & (Q_labels is not None):
                if np.any(P_labels[i] != Q_labels[j]):
                    continue
            elif P_labels is not None:
                if np.any(P_labels[i] != P_labels[i][0]):
                    continue
            elif Q_labels is not None:
                if np.any(Q_labels[j] != Q_labels[j][0]):
                    continue

            if (transform == 'rigid') & scale_invariant:
                scale = np.sqrt(P_scales[i] ** 2 + Q_scales[j] ** 2)
                p_scaled = p / scale
                q_scaled = q / scale
            elif (transform == 'rigid') & (not scale_invariant):
                p_scaled = p
                q_scaled = q

            elif (transform == 'similarity') & scale_invariant:
                p_scaled = p / P_scales[i]
                q_scaled = q / Q_scales[j]

            elif (transform == 'similarity') & (not scale_invariant):
                p_scaled = p
                q_scaled = q * P_scales[i] / Q_scales[j]

            else:
                raise RuntimeError('transform must be "rigid" or "similarity"')

            # import matplotlib.pyplot as plt
            # plt.plot(*p[0].T,'ro')
            # plt.plot(*p.T)
            # plt.plot(*q[0].T, 'ro')
            # plt.plot(*q.T)
            # plt.show()

            rmsd[i, j] = rmsd_kabsch(p_scaled, q_scaled)

    return rmsd
