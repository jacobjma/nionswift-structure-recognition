from collections import defaultdict
import numba
import numpy as np
import scipy.spatial


@numba.njit
def set_difference(ar1, ar2):
    mask = np.full(len(ar1), True)
    for a in ar2:
        mask &= (ar1 != a)
    return ar1[mask]


@numba.njit
def triangle_angles(p, r, q):
    a = np.sqrt(np.sum((r - q) ** 2, axis=1))
    b = np.sqrt(np.sum((q - p) ** 2, axis=1))
    c = np.sqrt(np.sum((p - r) ** 2, axis=1))
    a2 = a ** 2
    b2 = b ** 2
    c2 = c ** 2
    angles = np.zeros((len(p), 3))
    angles[:, 0] = np.arccos((b2 + c2 - a2) / (2 * b * c))
    angles[:, 1] = np.arccos((a2 + c2 - b2) / (2 * a * c))
    angles[:, 2] = np.pi - angles[:, 0] - angles[:, 1]
    return angles


@numba.njit
def stable_delaunay_cluster(points, simplices, neighbors, simplex_in_hull, threshold):
    angles = triangle_angles(points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]])
    labels = np.full(len(simplices), -1, np.int32)

    # assign all labels to faces in clustered with the outer face of the triangulation
    for i in simplex_in_hull:
        if labels[i] == -1:
            alpha = np.inf
            for j in np.where(neighbors[i] == -1)[0]:
                alpha = min(np.pi - angles[i][j], alpha)

            if alpha < threshold:
                queue = [i]
                labels[i] = 0
                while queue:
                    i = queue.pop()
                    for j in neighbors[i]:
                        if j != -1:
                            if labels[j] == -1:
                                k = set_difference(simplices[i], simplices[j])
                                l = set_difference(simplices[j], simplices[i])
                                alpha = np.pi - (angles[i][simplices[i] == k][0] + angles[j][simplices[j] == l][0])

                                if alpha < threshold:
                                    labels[j] = 0

                                    queue += [j]

    # assign labels to all other faces
    max_label = 1
    for i in range(0, len(simplices)):
        if labels[i] == -1:
            labels[i] = max_label
            queue = [i]
            while queue:
                i = queue.pop()
                for j in neighbors[i]:
                    if j != -1:
                        if labels[j] == -1:
                            k = set_difference(simplices[i], simplices[j])
                            l = set_difference(simplices[j], simplices[i])
                            alpha = np.pi - (angles[i][simplices[i] == k][0] + angles[j][simplices[j] == l][0])

                            if alpha < threshold:
                                labels[j] = max_label
                                queue += [j]

            max_label += 1

    return labels


def flatten(lists):
    return [item for sublist in lists for item in sublist]


def _directed_simplex_edges(simplices):
    edges = [[(simplex[i - 1], simplex[i]) for i in range(3)] for simplex in simplices]
    return flatten(edges)


def _order_exterior_vertices(simplices):
    edges = _directed_simplex_edges(simplices)
    tally = defaultdict(list)
    for i, item in enumerate(edges):
        tally[tuple(sorted(item))].append(i)

    edges = {edges[locs[0]][0]: edges[locs[0]][1] for locs in tally.values() if len(locs) == 1}

    order = [list(edges.keys())[0]]
    while len(order) < len(edges):
        order += [edges[order[-1]]]

    return order


def stable_delaunay_graph(points, threshold):
    delaunay = scipy.spatial.Delaunay(points)
    simplices = delaunay.simplices
    neighbors = delaunay.neighbors

    in_hull = np.where(np.any(neighbors == -1, axis=1))[0]

    labels = stable_delaunay_cluster(points, simplices, neighbors, in_hull, threshold)

    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(1, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')

    outer = set(list(simplices[labels == 0].flatten()))

    faces = []
    for i, (l, h) in enumerate(zip(lo, hi)):
        order = _order_exterior_vertices(simplices[np.sort(indices[l:h])])
        if not (set(order) & outer):
            faces += [order]

    # adjacency = [set() for _ in range(len(points))]
    # for face in faces:
    #    for i, j in enumerate(face):
    #        adjacency[j].add(face[i - 1])

    return faces
