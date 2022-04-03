import scipy
from collections import defaultdict
import numpy as np


def triangle_angles(p, r, q):
    angles = np.zeros((len(p), 3))
    side_lenghts = np.zeros((len(p), 3))

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


def adjacency_to_edges(adjacency):
    edges = []
    for i, adjacent in adjacency.items():
        for j in adjacent:
            edges.append([i, j])
    return edges


def edges_to_adjacency(edges):
    adjacency = defaultdict(list)
    for edge in edges:
        edge = list(edge)
        adjacency[edge[0]].append(edge[1])
        adjacency[edge[1]].append(edge[0])
    return adjacency


def order_adjacency_clockwise(points, adjacency, counter_clockwise=False):
    for node, adjacent in adjacency.items():
        centered = points[adjacent] - points[node]
        order = np.arctan2(centered[:, 0], centered[:, 1])
        adjacency[node] = [x for _, x in sorted(zip(order, adjacent), reverse=counter_clockwise)]
    return adjacency


def stable_delaunay_edges(points, threshold):
    delaunay = scipy.spatial.Delaunay(points)
    simplices = delaunay.simplices
    delaunay.close()

    angles = triangle_angles(points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]])
    angles = np.roll(angles, 2, axis=1).ravel()
    edges = np.stack([simplices, np.roll(simplices, 1, axis=1)], axis=-1).reshape((-1, 2))
    mask = angles < threshold
    edges = edges[mask]
    return set(frozenset(edge) for edge in edges)


def knn_edges(points, k):
    tree = scipy.spatial.KDTree(points)
    distances, indices = tree.query(points, k + 1)
    adjacency = {node: list(adjacent[1:]) for node, adjacent in enumerate(indices)}
    return set(frozenset(edge) for edge in adjacency_to_edges(adjacency))


def find_face(seed, adjacency, max_edges):
    i, j = seed, adjacency[seed][0]
    face = []
    for _ in range(max_edges):
        adjacent = list(adjacency[i]) + list(adjacency[i])
        i, j = adjacent[adjacent.index(j) + 1], i
        face.append(i)
        if i == seed:
            return face
    return None


def find_faces(adjacency, max_edges, min_edges=None):
    seeds = adjacency.keys()
    faces = []
    faces_set = set()
    for seed in seeds:
        face = find_face(seed, adjacency, max_edges)
        if face is None:
            continue

        if frozenset(face) not in faces_set:
            faces_set.add(frozenset(face))
            faces.append(face)

    if min_edges:
        return [face for face in faces if len(face) >= min_edges]
    else:
        return faces


def polygon_area(points):
    return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))


def faces_to_edges(faces):
    edges = set()
    for face in faces:
        if len(face) < 2:
            continue

        for i in range(len(face)):
            edges.add(frozenset({face[i - 1], face[i]}))

    return [list(edge) for edge in edges]


class RealSpaceCalibrator:

    def __init__(self,
                 model,
                 template,
                 lattice_constant,
                 min_sampling,
                 max_sampling,
                 step_size=.01,
                 binning=1):
        """
        Determine the sampling (pixel size) of an image using a neural network. The calibration works by sweeping over
        a set of candiate samplings and using the candidate with the best prediction to calculate the sampling using
        the detected bond lengths of the lattice.

        :param model: AtomRecognitionModel
            A neural network model for detecting atomic positions.
        :param template: str
            The lattice template. Currently only implemented for hexagonal lattices.
        :param lattice_constant: float
            The lattice constant in Ångstrom.
        :param min_sampling: float
            The minimum sampling of the sweep [Ångstrom / pixel].
        :param max_sampling: float
            The maximum sampling of the sweep [Ångstrom / pixel].
        :param step_size: float
            The step size of the sweep [Ångstrom / pixel]. Default is 0.1.
        :param binning: int
            Optional binning of the image prior to calculating predictions with the neural network. This will improve
            the speed of the predictions.
        """

        self.model = model
        self.template = template
        self.lattice_constant = lattice_constant
        self.min_sampling = min_sampling
        self.max_sampling = max_sampling
        self.step_size = step_size
        self.binning = binning

    def __call__(self, image):
        num_steps = int(np.ceil((self.max_sampling - self.min_sampling) / self.step_size))
        samplings = np.linspace(self.min_sampling, self.max_sampling, num_steps)

        if self.binning != 1.:
            image = scipy.ndimage.zoom(image, (1 / self.binning,) * 2)

        best_area = 0.
        best_sampling = None
        for sampling in samplings:
            points = self.model(image, sampling)['points']

            if len(points) < 4:
                continue

            edges = stable_delaunay_edges(points, .9)
            edges = edges.intersection(knn_edges(points, 3))

            if len(edges) < 3:
                continue

            edges = [list(edge) for edge in edges]
            adjacency = order_adjacency_clockwise(points, edges_to_adjacency(edges))

            faces = find_faces(adjacency, 6, 3)

            if len(faces) < 1:
                continue

            area = sum([polygon_area(points[face]) for face in faces])

            if area > best_area:
                best_area = area
                edges = points[np.array(faces_to_edges(faces))]

                bond_lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
                mean_bond_length = bond_lengths.mean()

                best_sampling = self.lattice_constant / np.sqrt(3) / mean_bond_length / self.binning

        return best_sampling
