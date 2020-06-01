import bisect
from collections import Counter

import numpy as np

from .utils import repeat, wrap
from ..construct import stable_delaunay_faces
from ..geometry import points_in_polygon, polygon_area, regular_polygon, kabsch_orientation
from ..representation import faces_to_adjacency, order_adjacency_clockwise
from ..rmsd import pairwise_rmsd
from ..select import select_faces_around_nodes
from ..transform import apply_affine_transform


def build_rectangular_graphene(a=2.46, n=1, m=1):
    basis = np.array([(0, 0), (2 / 3., 1 / 3.)])
    cell = np.array([[a, 0], [-a / 2., a * 3 ** 0.5 / 2.]])
    points = np.dot(basis, cell)
    points, cell = repeat(points, cell, 1, 2)
    cell = np.array([[cell[0, 0], 0], [0, cell[1, 1]]])
    points, cell = repeat(points, cell, n, m)
    return wrap(points, cell), cell


def outer_traversal_steps(defect_graph):
    dual_path = defect_graph.dual().outer_faces()[0]
    path = set(defect_graph.outer_faces()[0])
    steps = []
    for face in dual_path:
        steps.append(len(path.intersection(defect_graph.faces[face])))
    return steps


def graphene_reference_path(defect_graph):
    steps = outer_traversal_steps(defect_graph)
    a = np.array([1, 0])
    b = np.array([-1 / 2., 3 ** 0.5 / 2.])
    vectors = np.array([a, -b, -a - b, - a, b, a + b])
    k = 0
    reference_path = [vectors[k]]
    for i in range(len(steps)):
        j = (k - steps[i] + 3) % len(vectors)
        reference_path.append(reference_path[-1] + vectors[j])
        k = j
    return np.array(reference_path)


def defect_fingerprint(graph, shortened=False):
    dual = graph.dual()
    path = dual.outer_faces()[0]

    if not np.all([len(graph.faces[node]) == 6 for node in path]):
        raise RuntimeError('Defect is not enclosed')

    reference_path = graphene_reference_path(graph)

    if np.all(reference_path[0] != reference_path[-1]):
        raise RuntimeError('Defect is not enclosed')

    num_atoms_reference = polygon_area(reference_path) * 6 / (3 * np.sqrt(3) / 2)

    inside = points_in_polygon(graph.points, dual.points[path], return_indices=True)
    dual_inside = list(set(range(len(dual))) - set(path))

    degrees = Counter([graph.degrees[node] for node in inside])
    dual_degrees = Counter([dual.degrees[node] for node in dual_inside])

    num_atoms = len(inside)
    missing_atoms = int(np.round(num_atoms_reference - num_atoms))

    name = '{}.'.format(missing_atoms)

    if graph.labels is None:
        name += '0.'.format(np.sum(graph.labels[inside] != 1))
    else:
        name += '{}.'.format(np.sum(graph.labels[inside] != 1))

    name += '_'.join(['{}'.format('-'.join([str(key)] * dual_degrees[key])) for key in sorted(dual_degrees)]) + '.'

    if shortened:
        name += '{}'.format(num_atoms)
    else:
        name += '_'.join(['{}'.format('-'.join([str(key)] * degrees[key])) for key in sorted(degrees)])
    return name


def trace_clockwise_path(seed, clockwise_steps, adjacency, rotation=0):
    path = [seed]
    k = rotation
    for n in clockwise_steps:
        adjacent = np.array(adjacency[path[-1]])
        j = (k - n + 3) % len(adjacent)
        k = j
        path.append(adjacent[j])
    return path


def insert_defect_graphene(graph, defect_graph, seed, rotation=0):
    dual = graph.dual()

    insertion_steps = outer_traversal_steps(defect_graph)
    insertion_path_indices = trace_clockwise_path(seed, insertion_steps, dual.adjacency, rotation)

    if not np.all(np.array(dual.degrees)[insertion_path_indices] == 6):
        raise RuntimeError()

    insertion_path_points = dual.points[insertion_path_indices]

    nodes_to_delete = points_in_polygon(graph.points, insertion_path_points, return_indices=True)
    faces_to_delete = select_faces_around_nodes(nodes_to_delete, graph.faces)
    # stitch_faces = [graph.faces[i] for i in insertion_path_indices]
    graph, new_order = graph.delete_faces(faces_to_delete, True)

    defect_dual = defect_graph.dual()
    defect_dual_outer_face = defect_dual.outer_faces()[0]
    defect_dual_outer_polygon = defect_dual.outer_face_polygons()[0]

    defect_graph = defect_graph.copy()
    defect_graph.points = apply_affine_transform(defect_graph.points, src=defect_dual_outer_polygon,
                                                 dst=insertion_path_points[:-1])

    # defect_stitch_faces = [defect_graph.faces[i] for i in defect_dual_outer_face]
    defect_graph, defect_new_order = defect_graph.delete_faces(defect_dual_outer_face, True)

    # stitch_faces = [[new_order[node] if node in new_order.keys() else -1 for node in face] for face in stitch_faces]
    # defect_stitch_faces = [
    #     [defect_new_order[node] + len(graph) if node in defect_new_order.keys() else -1 for node in face] for face in
    #     defect_stitch_faces]

    graph.append(defect_graph)

    # for stitch_face, defect_stitch_face in zip(stitch_faces, defect_stitch_faces):
    #     stitch_face = deque(stitch_face)
    #     while (stitch_face[0] == -1) or (stitch_face[-1] != -1):
    #         stitch_face.rotate(-1)
    #
    #     defect_stitch_face = deque(defect_stitch_face)
    #     while (defect_stitch_face[0] == -1) or (defect_stitch_face[-1] != -1):
    #         defect_stitch_face.rotate(-1)
    #
    #     stitch_face = list(filter(lambda x: x != -1, stitch_face))
    #     defect_stitch_face = list(filter(lambda x: x != -1, defect_stitch_face))
    #     graph._faces += [stitch_face + defect_stitch_face]

    return graph


def neighbor_template(bond_length):
    template = np.zeros((4, 2))
    template[1:] = regular_polygon(bond_length * np.sqrt(3), 3)
    return template


def neighbor_segments(adjacency, points):
    segments = []
    for node, adjacent in adjacency.items():
        segment = np.vstack(([points[node]], points[adjacent])).astype(np.float)
        segment -= np.mean(segment, axis=0)
        segments.append(segment)
    return segments


def assign_sublattice(points, bond_length, alpha=2.2, principal_orientation=None):
    faces = stable_delaunay_faces(points, alpha)

    adjacency = faces_to_adjacency(faces, len(points))
    adjacency = order_adjacency_clockwise(points, adjacency, True)

    segments = neighbor_segments(adjacency, points)
    templates = [neighbor_template(bond_length)]

    rmsd = pairwise_rmsd(templates, segments).ravel()

    node = rmsd.argmin()

    sublattice = np.full(len(adjacency), -1)

    src, dst = segments[node], templates[0]
    orientation = kabsch_orientation(src, dst)

    orientation = orientation % (2 * np.pi / 3)

    sublattice[node] = 0
    if principal_orientation:
        if np.abs(orientation - principal_orientation) > (np.pi / 6):
            sublattice[node] = 1
            orientation = orientation % (np.pi / 3)

    queue = [(rmsd, node)]

    while queue:
        _, node = queue.pop(0)
        neighbors = np.array(adjacency[node])
        neighbors = neighbors[sublattice[neighbors] == -1]
        sublattice[neighbors] = sublattice[node] == 0
        for neighbor in neighbors:
            bisect.insort(queue, (rmsd[neighbor], neighbor))

    return sublattice, orientation
