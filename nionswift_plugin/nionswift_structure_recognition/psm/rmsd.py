import numba as nb
import numpy as np
import scipy


def rmsd_kabsch(p, q):
    """ The minimized RMSD between two sets of points with Kabsch algorithm. """

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
    """ The minimized RMSD between two sets of points with the QCP algorithm.
    """

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

    # Select relevant columns, depending on params
    A = A[:, list(coeffs) + [8]]

    _, _, V = np.linalg.svd(A)

    H = np.zeros((3, 3))
    # solution is right singular vector that corresponds to smallest
    # singular value
    H.flat[list(coeffs) + [8]] = - V[-1, :-1] / V[-1, -1]
    H[2, 2] = 1

    return H


class RMSD(object):

    def __init__(self, transform='similarity', rotation_invariant=True, scale_invariant=True, pivot='cop'):

        if transform not in ['rigid', 'similarity']:
            raise ValueError()

        if pivot not in ['cop', 'front']:
            raise ValueError()

        self.transform = transform
        self.scale_invariant = scale_invariant
        self.rotation_invariant = rotation_invariant
        self.pivot = pivot

        self._points = None
        self._rmsd = None

    def register(self, points, segments, templates, labels=None, calculate_strain=False, progress_bar=False):

        self._rmsd = np.empty((len(segments), len(templates)))
        self._rmsd[:] = np.inf
        self._permutations = np.zeros((len(segments), len(templates)), np.int)
        self._points = points
        self._segments = segments
        self._templates = templates

        for i, segment in enumerate(segments):
            p = points[segment]

            if labels is not None:
                p_labels = labels[segment]
            else:
                p_labels = None

            if self.pivot == 'cop':
                p = p - np.mean(p, axis=0)

            else:
                p = p - p[0]

            p_scale = np.sqrt(np.sum(np.linalg.norm(p, axis=1) ** 2))

            if (self.transform == 'similarity') & self.scale_invariant:
                p = p / p_scale

            for j, template in enumerate(templates):
                best_rmsd = np.inf

                if len(p) == len(template):

                    for k, (q, q_labels) in enumerate(template.generate_permutations()):
                        if np.all(q_labels == p_labels):
                            q_scale = template.scale

                            if (self.transform == 'rigid') & self.scale_invariant:
                                scale = np.sqrt(p_scale ** 2 + q_scale ** 2)
                                p = p / scale
                                q = q / scale

                            elif (self.transform == 'similarity') & self.scale_invariant:
                                q = q / q_scale

                            elif (self.transform == 'similarity') & (not self.scale_invariant):
                                q = q * p_scale / q_scale

                            # if i==45:
                            #     import matplotlib.pyplot as plt
                            #     plt.plot(*p.T)
                            #     plt.plot(*q.T)
                            #     for k,p_ in enumerate(p):
                            #        plt.annotate('{}'.format(k),xy=p_)
                            #     for k,q_ in enumerate(q):
                            #        plt.annotate('{}'.format(k),xy=q_)
                            #     plt.show()

                            rmsd = rmsd_qcp(p, q)

                            if rmsd < best_rmsd:
                                self._rmsd[i, j] = rmsd_qcp(p, q)
                                self._permutations[i, j] = k
                                best_rmsd = self._rmsd[i, j]

                                # import matplotlib.pyplot as plt
                                # plt.plot(*p.T)
                                # plt.plot(*q.T)
                                # for k,p_ in enumerate(p):
                                #    plt.annotate('{}'.format(k),xy=p_)
                                # for k,q_ in enumerate(q):
                                #    plt.annotate('{}'.format(k),xy=q_)
                                # plt.show()

        return self._rmsd

    def best_rmsd(self):
        rmsd = self._rmsd.copy()

        best_matches = -np.ones(len(rmsd), dtype=int)

        valid = np.any(rmsd < np.inf, axis=1)

        best_matches[valid] = np.nanargmin(rmsd[valid, :], axis=1)

        return rmsd[range(len(best_matches)), best_matches], best_matches

    def calculate_strain(self, rmsd_max=np.inf):
        strain = np.zeros((len(self._segments), 2, 2))
        rotation = np.zeros((len(self._segments), 1))

        strain[:] = np.nan
        rotation[:] = np.nan

        rmsd, best_match = self.best_rmsd()

        for i, segment in enumerate(self._segments):

            if rmsd[i] < rmsd_max:
                p = self._points[segment]
                template = self._templates[best_match[i]]

                if template.permutations is None:
                    q = template.points
                else:
                    permutation = template.permutations[self._permutations[i, best_match[i]]]

                    #print(permutation)
                    q = template.points[permutation]

                # print(p,q)

                A = affine_transform(p, q)

                # if return_affine:
                #    affine[i] = A
                # else:
                U, P = scipy.linalg.polar(A[:-1, :-1], side='left')

                rotation[i] = np.arctan2(U[0, 1], U[0, 0])
                strain[i] = P - np.identity(2)

        return strain, rotation
