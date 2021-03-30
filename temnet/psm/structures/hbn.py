import numpy as np

from temnet.psm.coloring import prioritized_greedy_two_coloring
from temnet.psm.dual import faces_to_dual_adjacency
from temnet.psm.rmsd import batch_rmsd_qcp
from temnet.psm.template import regular_polygon
from temnet.psm.traversal import connected_components
from temnet.psm.utils import is_point_in_polygon, remove_repetition_cyclical, faces_to_double_edge, faces_to_quad_edge
from temnet.psm.utils import order_adjacency_clockwise, flatten_list_of_lists, outer_faces_from_faces, mask_nodes

from temnet.psm.graph import GeometricGraph


def hbn_reference_counts(reference_path, first_label_to_the_left):
    a = np.array([1, 0])
    b = np.array([-1 / 2., 3 ** 0.5 / 2.])

    new_reference_path = np.dot(reference_path, np.linalg.inv(np.array([a, b])))

    x = np.arange(new_reference_path[:, 0].min() - 1, new_reference_path[:, 0].max() + 2)
    y = np.arange(new_reference_path[:, 1].min() - 1, new_reference_path[:, 1].max() + 2)

    x, y = np.meshgrid(x, y)
    reference_lattice = np.array([x.ravel(), y.ravel()]).T
    reference_lattice = np.dot(reference_lattice, np.array([a, b]))

    basis = np.array([[0, 1 / np.sqrt(3)], [0, -1 / np.sqrt(3)]])

    reference_lattice = reference_lattice[None] + basis[:, None]
    reference_lattice = reference_lattice.reshape((-1, 2))

    if first_label_to_the_left == 1:
        reference_labels = np.zeros(len(reference_lattice))
        reference_labels[:reference_labels.shape[0] // 2] = 1
    else:
        reference_labels = np.ones(len(reference_lattice))
        reference_labels[:reference_labels.shape[0] // 2] = 0

    is_in_defect = [is_point_in_polygon(point, reference_path) for point in reference_lattice]

    return (reference_labels[is_in_defect] == 0).sum(), (reference_labels[is_in_defect] == 1).sum()


def step_clockwise_in_hexagonal_lattice(steps):
    a = np.array([1, 0])
    b = np.array([-1 / 2., 3 ** 0.5 / 2.])
    vectors = np.array([a, -b, -a - b, - a, b, a + b])
    k = 0
    reference_path = [vectors[k]]
    for i in range(len(steps)):
        j = (k + steps[i] + 3) % len(vectors)
        reference_path.append(reference_path[-1] + vectors[j])
        k = j

    return np.array(reference_path)


def assign_sublattice_hexagonal(adjacency, points, lattice_constant):
    template = regular_polygon(3, lattice_constant, append_center=True)
    segments = [[node] + adjacent for node, adjacent in adjacency.items()]
    three_connected = np.array([len(segment) == 4 for segment in segments])

    rmsd = np.zeros(len(segments))
    rmsd[three_connected == 0] = np.inf
    segments = points[np.array([segment for i, segment in enumerate(segments) if three_connected[i]])]
    segments -= segments[:, 0, None]

    rmsd[three_connected] = batch_rmsd_qcp(template[None], segments)
    labels = prioritized_greedy_two_coloring(adjacency, rmsd)
    return labels


def connected_equalsized_faces(faces, face_centers, size, negate=False, discard_outer=True, return_boundaries=False):
    if negate:
        equal_adjacent_nodes = np.array([len(face) for face in faces]) != size
    else:
        equal_adjacent_nodes = np.array([len(face) for face in faces]) == size

    if discard_outer:
        outer = flatten_list_of_lists(outer_faces_from_faces(faces))
        outer_adjacent = [True if not set(face).intersection(outer) else False for face in faces]
        equal_adjacent_nodes *= outer_adjacent

    dual_adjacency = order_adjacency_clockwise(face_centers, faces_to_dual_adjacency(faces))
    components = connected_components(mask_nodes(dual_adjacency, equal_adjacent_nodes))

    if return_boundaries:
        return [outer_faces_from_faces([faces[x] for x in component])[0] for component in components]
    else:
        return components


def hbn_defect_metrics(points, labels, faces, defect_boundaries):
    face_centers = np.array([points[face].mean(0) for face in faces])
    double_edge = faces_to_double_edge(faces)
    reverse_quad_edge = {tuple(value): key for key, value in faces_to_quad_edge(faces).items()}

    metrics = []
    for boundary in defect_boundaries:
        path = []
        for i, j in zip(boundary, np.roll(boundary, -1)):
            path.append(double_edge[(i, j)])

        path, repetitions = remove_repetition_cyclical(path)
        first_label_to_the_left = reverse_quad_edge[(path[0], path[1])][0]
        reference_path = step_clockwise_in_hexagonal_lattice(np.array(repetitions) + 1)
        reference_count = hbn_reference_counts(reference_path, labels[first_label_to_the_left])

        is_in_defect = [is_point_in_polygon(point, face_centers[path]) for point in points]
        unique, count = np.unique(labels[is_in_defect], return_counts=True)

        print(unique, count)

        missing = (reference_count[0] - count[0], reference_count[1] - count[1])

        center = points[path].mean(0)

        metrics.append({'boundary': boundary,
                        'enclosing_path': path,
                        'center': center,
                        'num_missing': missing})

    return metrics



