import numba as nb
import numpy as np
import scipy.spatial
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph

from .utils import generate_indices, labels_to_lists
from .representation import adjacency_to_matrix


def triangle_angles(p, r, q):
    angles = np.zeros((len(p), 3))

    a2 = np.sum((r - q) ** 2, axis=1)
    b2 = np.sum((q - p) ** 2, axis=1)
    c2 = np.sum((p - r) ** 2, axis=1)

    a = np.sqrt(a2)
    b = np.sqrt(b2)
    c = np.sqrt(c2)

    A = (b2 + c2 - a2) / (2 * b * c)
    A[A > 1] = 1
    A[A < -1] = -1
    angles[:, 0] = np.arccos(A)

    B = (a2 + c2 - b2) / (2 * a * c)
    B[B > 1] = 1
    B[B < -1] = -1
    angles[:, 1] = np.arccos(B)

    angles[:, 2] = np.pi - angles[:, 0] - angles[:, 1]
    return angles


def delaunay_simplex_distance_metrics(points, simplices, neighbors):
    print(neighbors)
    angles, side_lengths = triangle_angles(points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]])
    alpha = np.zeros(len(neighbors) * 6, dtype=np.float64)
    row_ind = np.zeros(len(neighbors) * 6, dtype=np.int32)
    col_ind = np.zeros(len(neighbors) * 6, dtype=np.int32)

    l = 0
    for i in range(len(neighbors)):
        for j in range(3):
            k = neighbors[i][j]
            if k == -1:
                row_ind[l] = i
                col_ind[l] = len(neighbors)
                row_ind[l + 1] = len(neighbors)
                col_ind[l + 1] = i

                alpha[l] = alpha[l + 1] = angles[i][j]

            else:
                row_ind[l] = i
                col_ind[l] = k
                row_ind[l + 1] = k
                col_ind[l + 1] = i

                for m in range(3):
                    if not np.any(simplices[k][m] == simplices[i]):
                        break

                alpha[l] = alpha[l + 1] = angles[i][j] + angles[k][m]
            l += 2

    return alpha, (row_ind, col_ind)


def directed_simplex_edges(simplices):
    edges = np.zeros((len(simplices) * 3, 2), dtype=np.int64)
    k = 0
    for i in range(len(simplices)):
        for j in range(3):
            edges[k][0] = simplices[i][j - 1]
            edges[k][1] = simplices[i][j]
            k += 1

    return edges[:k]


def order_exterior_vertices(edges):
    boundary = dict()

    for i in range(len(edges)):
        edge = edges[i]
        if not np.any(np.sum(edge[::-1] == edges[:], axis=1) == 2):
            boundary[edge[0]] = edge[1]

    order = [boundary[list(boundary.keys())[0]]]
    for i in range(len(boundary) - 1):
        order.append(boundary[order[i]])

    return order


def join_simplices(simplices, labels):
    joined_simplices = []
    for i, simplex_indices in generate_indices(labels):
        if len(simplex_indices) > 0:
            simplex_edges = directed_simplex_edges(simplices[simplex_indices])
            joined_simplices.append(order_exterior_vertices(simplex_edges))
    return joined_simplices


def stable_delaunay_faces(points, alpha_threshold, r_threshold=np.inf):
    delaunay = scipy.spatial.Delaunay(points)
    simplices = delaunay.simplices
    neighbors = delaunay.neighbors
    delaunay.close()

    alpha, (row_ind, col_ind) = delaunay_simplex_distance_metrics(points, simplices, neighbors)

    connected = (alpha > alpha_threshold) + (r > r_threshold)

    row_ind = row_ind[connected]
    col_ind = col_ind[connected]

    M = csr_matrix((np.ones(len(row_ind), dtype=np.bool), (row_ind, col_ind)), (len(neighbors) + 1,) * 2, dtype=np.bool)

    _, labels = scipy.sparse.csgraph.connected_components(M)
    labels[labels == labels[-1]] = -1

    faces = join_simplices(simplices, labels)
    return faces


def connected_components(adjacency):
    M = adjacency_to_matrix(adjacency)
    _, labels = scipy.sparse.csgraph.connected_components(M)
    return labels_to_lists(labels, list(adjacency.keys()))
