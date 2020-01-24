import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy.spatial
from sklearn.cluster import DBSCAN

from .graphutils import adjacency_to_faces, outer_face_from_inner
from .graphutils import faces_to_adjacency, adjacency_to_edges, dual_faces, linegraph_faces, order_looping_edges
from .segments import RunLengthSegments, segment_centers
from .utils import generate_indices, set_difference, check_clockwise, simplex_circumcenter
from .visualize.mpl import add_edges


@nb.njit
def triangle_angles(p, r, q):
    a = np.sqrt(np.sum((r - q) ** 2, axis=1))
    b = np.sqrt(np.sum((q - p) ** 2, axis=1))
    c = np.sqrt(np.sum((p - r) ** 2, axis=1))
    a2 = a ** 2
    b2 = b ** 2
    c2 = c ** 2
    angles = np.zeros((len(p), 3))
    x = (b2 + c2 - a2) / (2 * b * c)
    x[x > 1] = 1
    x[x < -1] = -1
    angles[:, 0] = np.arccos(x)
    x = (a2 + c2 - b2) / (2 * a * c)
    x[x > 1] = 1
    x[x < -1] = -1
    angles[:, 1] = np.arccos(x)
    angles[:, 2] = np.pi - angles[:, 0] - angles[:, 1]
    return angles


@nb.njit
def _stable_delaunay_cluster(points, simplices, neighbors, threshold, length_threshold):
    angles = triangle_angles(points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]])
    labels = np.full(len(simplices), -2, np.int32)

    in_hull = np.where(np.sum(neighbors == -1, axis=1) > 0)[0]

    for i in in_hull:
        if labels[i] == -2:
            alpha = np.inf
            d = 0.
            for j in np.where(neighbors[i] == -1)[0]:
                alpha = min(np.pi - angles[i][j], alpha)

            for j in np.where(neighbors[i] != -1)[0]:
                d = max(np.linalg.norm(points[simplices[i][j - 1]] - points[simplices[i][j]]), d)

            if (alpha < threshold) | (d > length_threshold):
                queue = [i]
                labels[i] = -1
                while queue:
                    i = queue.pop()
                    for j in neighbors[i]:
                        if j != -1:
                            if labels[j] == -2:
                                k = set_difference(simplices[i], simplices[j])
                                l = set_difference(simplices[j], simplices[i])
                                alpha = np.pi - (angles[i][simplices[i] == k][0] + angles[j][simplices[j] == l][0])

                                if alpha < threshold:
                                    labels[j] = -1
                                    queue += [j]

                                elif length_threshold < np.inf:
                                    edge = simplices[i][simplices[i] != k]
                                    if (np.linalg.norm(points[edge][0] - points[edge][1]) > length_threshold):
                                        labels[j] = -1
                                        queue += [j]

    # assign labels to all other faces
    max_label = 0
    for i in range(0, len(simplices)):
        if labels[i] == -2:
            labels[i] = max_label
            queue = [i]
            while queue:
                i = queue.pop()
                for j in neighbors[i]:
                    if j != -1:
                        if labels[j] == -2:
                            k = set_difference(simplices[i], simplices[j])
                            l = set_difference(simplices[j], simplices[i])
                            alpha = np.pi - (angles[i][simplices[i] == k][0] + angles[j][simplices[j] == l][0])

                            if alpha < threshold:
                                labels[j] = max_label
                                queue += [j]

                            elif length_threshold < np.inf:
                                edge = simplices[i][simplices[i] != k]
                                if (np.linalg.norm(points[edge][0] - points[edge][1]) > length_threshold):
                                    labels[j] = max_label
                                    queue += [j]

            max_label += 1
    return labels


@nb.njit
def stable_delaunay_distance(points, simplices, neighbors):
    angles = triangle_angles(points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]])
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

                alpha[l] = alpha[l + 1] = np.pi - angles[i][j]
                l += 2
            else:
                row_ind[l] = i
                col_ind[l] = k
                row_ind[l + 1] = k
                col_ind[l + 1] = i

                for m in range(3):
                    if not np.any(simplices[k][m] == simplices[i]):
                        break

                alpha[l] = alpha[l + 1] = np.pi - (angles[i][j] + angles[k][m])
                l += 2

    return alpha[:l], (row_ind[:l], col_ind[:l])


@nb.njit
def directed_simplex_edges(simplices):
    edges = np.zeros((len(simplices) * 3, 2), dtype=np.int32)
    k = 0
    for i in range(len(simplices)):
        for j in range(3):
            edges[k][0] = simplices[i][j - 1]
            edges[k][1] = simplices[i][j]
            k += 1

    return edges[:k]


@nb.njit
def order_exterior_vertices(simplices):
    edges = directed_simplex_edges(simplices)

    boundary = nb.typed.Dict.empty(
        key_type=nb.types.int32,
        value_type=nb.types.int32,
    )
    for i in range(len(edges)):
        edge = edges[i]
        if not np.any(np.sum(edge[::-1] == edges[:], axis=1) == 2):
            boundary[edge[0]] = edge[1]

    order = np.zeros(len(boundary), dtype=np.int32)
    order[0] = boundary[list(boundary.keys())[0]]
    i = 0
    while i < len(boundary) - 1:
        order[i + 1] = boundary[order[i]]
        i += 1

    return order


def circumcenter_clustering(points, threshold):
    delaunay = scipy.spatial.Delaunay(points)
    simplices = delaunay.simplices

    vertices = np.array([simplex_circumcenter(*points[s]) for s in simplices])

    outer = order_looping_edges(delaunay.convex_hull)[:, 0]

    if check_clockwise(points[outer]):
        outer = outer[::-1]

    estimator = DBSCAN(eps=threshold, min_samples=2)

    labels = estimator.fit_predict(vertices)

    return join_simplex_clusters(simplices, labels)


# def stable_delaunay_graph(points, threshold):
#     delaunay = scipy.spatial.Delaunay(points)
#     simplices = delaunay.simplices
#     neighbors = delaunay.neighbors
#
#     data, (row_ind, col_ind) = stable_delaunay_distance(points, simplices, neighbors)
#
#     m = csr_matrix((data, (row_ind, col_ind)), (len(neighbors) + 1,) * 2)
#
#     estimator = DBSCAN(eps=threshold, min_samples=1, metric='precomputed')
#     labels = estimator.fit_predict(m)
#
#     indices, partitions = join_simplex_clusters(simplices, labels)
#     num_nodes = len(points)
#
#     outer = outer_face_from_inner(indices, partitions, num_nodes)[:, 0]
#
#     if check_clockwise(points[outer]):
#         outer = outer[::-1]
#
#     indices = np.append(indices, outer)
#     partitions = np.append(partitions, len(indices))
#
#     return indices, partitions


def stable_delaunay_graph(points, threshold, cutoff=np.inf):
    delaunay = scipy.spatial.Delaunay(points)
    simplices = delaunay.simplices
    neighbors = delaunay.neighbors

    labels = _stable_delaunay_cluster(points, simplices, neighbors, threshold, cutoff)

    indices, partitions = join_simplex_clusters(simplices, labels)
    num_nodes = len(points)

    outer = outer_face_from_inner(indices, partitions, num_nodes)

    for i, face in enumerate(outer):
        face = face[:, 0]
        if check_clockwise(points[face]):
            face = face[::-1]

        indices = np.append(indices, face)
        partitions = np.append(partitions, len(indices))

    return indices, partitions


def join_simplex_clusters(simplices, labels):
    indices = np.zeros(len(labels) * 4, dtype=np.int32)
    partitions = np.zeros(np.max(labels) + 3, dtype=np.int32)

    j = 0
    l = 0
    for i, simplex_indices in generate_indices(labels):
        order = order_exterior_vertices(simplices[np.sort(simplex_indices)])
        k = len(order)
        partitions[l] = j
        indices[j:j + k] = order
        j = j + k
        l += 1
    partitions[l] = j
    indices = indices[:j]
    return indices, partitions[:l + 1]


@nb.njit
def clockwise_traversal(indices, partitions, max_depth, start_nodes, roll_start=0):
    num_nodes = len(partitions) - 1
    degrees = np.diff(partitions)
    max_degree = np.max(degrees)

    traversal_indices = np.zeros(len(start_nodes) * max_degree ** max_depth, dtype=np.int32)
    traversal_partitions = np.zeros(len(start_nodes) + 1, dtype=np.int32)

    visited = np.zeros(num_nodes, dtype=np.bool_)
    depth = np.zeros(num_nodes, dtype=np.int32)
    incident = np.zeros(num_nodes, dtype=np.int32)

    m = 0
    for i in range(len(start_nodes)):
        visited[:] = False
        depth[:] = -1
        incident[:] = -1

        traversal_indices[m] = start_nodes[i]
        traversal_partitions[i] = m
        m += 1

        depth[start_nodes[i]] = 0
        depth_reached = False
        j = 0

        while not depth_reached:
            k = traversal_indices[traversal_partitions[i] + j]

            if not visited[k]:
                adjacent = indices[partitions[k]:partitions[k + 1]]

                if incident[k] > -1:
                    first = np.nonzero(adjacent == incident[k])[0][0]
                    adjacent = np.roll(adjacent, -first)[1:]
                elif roll_start > 0:
                    adjacent = np.roll(adjacent, roll_start)

                for l in adjacent:

                    if depth[l] == -1:

                        if depth[k] + 1 == max_depth:
                            depth_reached = True
                        else:
                            depth[l] = depth[k] + 1
                            incident[l] = k
                            traversal_indices[m] = l
                            m += 1

                visited[k] = 1

                if np.all(visited):
                    depth_reached = True

            j += 1

    traversal_partitions[-1] = m
    return traversal_indices[:m], traversal_partitions


@nb.njit
def order_adjacency_clockwise(points, indices, partitions, exclude_first=False):
    if exclude_first:
        m = 1
    else:
        m = 0

    for i in range(len(partitions) - 1):
        centered = points[indices[partitions[i] + m:partitions[i + 1]]] - points[i]
        order = np.argsort(-np.arctan2(centered[:, 0], centered[:, 1]))
        indices[partitions[i] + m:partitions[i + 1]] = indices[partitions[i] + m:partitions[i + 1]][order]
    return indices


class GeometricGraph(object):

    def __init__(self, points, faces=None, labels=None):
        if points is None:
            self._points = np.zeros((0, 2))
        else:
            self._points = points

        self._faces = faces
        self._labels = labels

    def __len__(self):
        return len(self.points)

    def build_stable_delaunay_graph(self, alpha, cutoff=np.inf):

        indices, partitions = stable_delaunay_graph(self.points, alpha, cutoff)
        self._faces = RunLengthSegments(indices, partitions)

    def build_circumcenter_cluster_graph(self, threshold):
        self._faces = RunLengthSegments(*circumcenter_clustering(self.points, threshold))

    @property
    def points(self):
        return self._points

    @property
    def num_nodes(self):
        return len(self.points)

    @property
    def labels(self):
        return self._labels

    def translate(self, amount):
        self._points = self._points + amount

    def edges(self):
        adjacency = self.adjacency(order_clockwise=False)
        return adjacency_to_edges(adjacency.indices, adjacency.partitions)

    def faces(self):
        return self._faces

    def traversals(self, max_depth, start_nodes=None, roll_start=0, max_degree=100):
        if start_nodes is None:
            start_nodes = np.arange(len(self))
        else:
            start_nodes = np.array(start_nodes).reshape((1,))

        adjacency = self.adjacency(max_degree=max_degree)
        return RunLengthSegments(
            *clockwise_traversal(adjacency.indices, adjacency.partitions, max_depth, start_nodes, roll_start))

    def adjacency(self, order_clockwise=True, include_self=False, max_degree=100):
        faces = self.faces()
        indices, partitions = faces_to_adjacency(faces.indices, faces.partitions, len(self), max_degree, include_self)
        if order_clockwise:
            indices = order_adjacency_clockwise(self.points, indices, partitions, include_self)
        return RunLengthSegments(indices, partitions)

    def faces_list(self):
        return [face for face in self.faces()]

    def face_centers(self):
        faces = self.faces()
        return segment_centers(self.points, faces.indices, faces.partitions)[:-1]

    def dual(self):
        faces = self.faces()
        centers = segment_centers(self.points, faces.indices, faces.partitions)[:-1]
        indices, partitions = dual_faces(self.points, faces.indices, faces.partitions)
        return self.__class__(centers, RunLengthSegments(indices, partitions), faces.sizes()[:-1])

    def linegraph(self):
        adjacency = self.adjacency()
        faces = self.faces()
        indices, fronts, centers, edge_labels = linegraph_faces(self.points, faces.indices, faces.partitions,
                                                                adjacency.indices, adjacency.partitions)
        faces = RunLengthSegments(indices, fronts)
        return self.__class__(centers, faces, edge_labels)

    def get_subgraph(self, node_indices):
        return SubGraph(self, node_indices)

    def show(self, ax=None, show_numbering=False, show_labels=False, node_size=100, colors=None, linewidths=1.):
        if ax is None:
            ax = plt.subplot()
        add_edges(ax, self.points, self.edges(), colors=colors, linewidths=linewidths)

        ax.scatter(*self.points.T, c='k', s=node_size)

        if show_labels & (self.labels is not None):
            for label, point in zip(self.labels, self.points):
                ax.annotate('{}'.format(label), xy=point, ha='center', va='center', color='w', size=12)

        if show_numbering:
            for i, point in enumerate(self.points):
                ax.annotate('{}'.format(i), xy=point, ha='center', va='center', color='w', size=12)

    def save(self, filename):
        save_graph(filename, self.points, self.faces(), self.labels)


def save_graph(filename, points, faces, labels=None):
    kwargs = {'points': points}

    if faces is not None:
        kwargs.update({'indices': faces.indices,
                       'partitions': faces.partitions,
                       })

    if labels is not None:
        kwargs.update({'labels': labels})

    np.savez(filename, **kwargs)


def load_graph(filename):
    npzfile = np.load(filename)

    points = npzfile['points']

    try:
        faces = RunLengthSegments(npzfile['indices'], npzfile['partitions'])
    except KeyError:
        faces = None

    try:
        labels = npzfile['labels']
    except KeyError:
        labels = None

    return GeometricGraph(points, faces, labels)


# @nb.njit
def subgraph_adjacency(indices, partitions, subgraph):
    subgraph_indices = np.zeros(indices.shape, dtype=indices.dtype)
    subgraph_partitions = np.zeros(len(subgraph) + 1, dtype=partitions.dtype)
    l = 0
    for i in range(len(subgraph)):
        subgraph_partitions[i] = l
        j = subgraph[i]
        adjacent = indices[partitions[j]:partitions[j + 1]]
        for k in adjacent:
            m = np.nonzero(k == subgraph)[0]
            if len(m) > 0:
                subgraph_indices[l] = m[0]
                l += 1
    subgraph_indices = subgraph_indices[:l]
    subgraph_partitions[-1] = l
    return subgraph_indices, subgraph_partitions


class SubGraph(object):

    def __init__(self, graph, indices):
        self._graph = graph
        self._indices = indices

    @property
    def points(self):
        return self.graph.points[self.indices]

    @property
    def labels(self):
        labels = self.graph.labels
        if labels is None:
            return None
        else:
            return labels[self.indices]

    @property
    def indices(self):
        return self._indices

    @property
    def graph(self):
        return self._graph

    def adjacency(self):
        adjacency = self.graph.adjacency()
        return RunLengthSegments(*subgraph_adjacency(adjacency.indices, adjacency.partitions, self.indices))

    def edges(self):
        adjacency = self.graph.adjacency(order_clockwise=False)
        indices, partitions = subgraph_adjacency(adjacency.indices, adjacency.partitions, self.indices)
        return adjacency_to_edges(indices, partitions)

    def detach(self):
        points = self.points
        adjacency = self.adjacency()
        faces = RunLengthSegments(*adjacency_to_faces(points, adjacency.indices, adjacency.partitions))
        return GeometricGraph(points, faces, self.labels)
