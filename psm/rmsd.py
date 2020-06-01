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


def pairwise_rmsd(A, B, A_labels=None, B_labels=None, transform='rigid', scale_invariant=True, pivot='cop'):

    if A_labels is not None:
        if len(A_labels) != len(A):
            raise RuntimeError()
        A_labels = [np.array(a_labels) for a_labels in A_labels]

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

    # k = 0
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if len(a) != len(b):
                continue

            if (A_labels is not None) & (B_labels is not None):
                if np.any(A_labels[i] != B_labels[j]):
                    continue
            elif A_labels is not None:
                if np.any(A_labels[i] != A_labels[i][0]):
                    continue
            elif B_labels is not None:
                if np.any(B_labels[j] != B_labels[j][0]):
                    continue

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

            # if k < 1:
            #     import matplotlib.pyplot as plt
            #     plt.plot(*a_.T)
            #     plt.plot(*b_.T)
            #     plt.plot(a_[0, 0], a_[0, 1], 'o')
            #     plt.plot(b_[0,0],b_[0,1],'o')
            #     plt.show()
            #     k+=1

            rmsd[i, j] = rmsd_qcp(a_, b_)

    return rmsd
