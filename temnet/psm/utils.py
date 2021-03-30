from collections import defaultdict

import numpy as np


def connect_edges(edges):
    def add_next_to_connected_edges(connected_edges, edges):
        found_next_edge = False
        for i, edge in enumerate(edges):
            if connected_edges[-1][-1] == edge[0]:
                connected_edges[-1].append(edge[1])
                found_next_edge = True
                del edges[i]
                break

            elif connected_edges[-1][-1] == edge[1]:
                connected_edges[-1].append(edge[0])
                found_next_edge = True
                del edges[i]
                break

        if found_next_edge == False:
            connected_edges.append([edges[0][1]])
            del edges[0]

        return connected_edges, edges

    connected_edges = [[edges[0][1]]]
    del edges[0]

    while edges:
        connected_edges, edges = add_next_to_connected_edges(connected_edges, edges)

    return connected_edges


def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def order_adjacency_clockwise(points, adjacency, counter_clockwise=False):
    for node, adjacent in adjacency.items():
        centered = points[adjacent] - points[node]
        order = np.arctan2(centered[:, 0], centered[:, 1])
        adjacency[node] = [x for _, x in sorted(zip(order, adjacent), reverse=counter_clockwise)]
    return adjacency


def faces_to_adjacency(faces, num_nodes):
    adjacency = defaultdict(set)
    for face in faces:
        for i in range(len(face)):
            adjacency[face[i]].add(face[i - 1])
            adjacency[face[i - 1]].add(face[i])

    return {i: list(adjacency[i]) for i in range(num_nodes)}


def faces_to_edges(faces):
    edges = set()
    for face in faces:
        if len(face) < 2:
            continue

        for i in range(len(face)):
            edge = frozenset({face[i - 1], face[i]})
            if len(edge) > 1:
                edges.add(edge)

    return [list(edge) for edge in edges]


def faces_to_quad_edge(faces):
    quad_edge = defaultdict(lambda: list([None, None]))
    for i, face in enumerate(faces):
        for j in range(len(face)):
            quad_edge[(face[j - 1], face[j])][0] = i
            quad_edge[(face[j], face[j - 1])][-1] = i
    return quad_edge


def faces_to_double_edge(faces):
    double_edge = defaultdict()
    for i, face in enumerate(faces):
        for j in range(len(face)):
            double_edge[(face[j - 1], face[j])] = i
    return double_edge


def edges_to_adjacency(edges):
    adjacency = defaultdict(list)
    for edge in edges:
        edge = list(edge)
        adjacency[edge[0]].append(edge[1])
        adjacency[edge[1]].append(edge[0])
    return adjacency


def adjacency_to_edges(adjacency):
    edges = []
    for i, adjacent in adjacency.items():
        for j in adjacent:
            edges.append([i, j])
    return edges


def check_clockwise(polygon):
    clockwise = False
    signed_area = 0.
    for i in range(len(polygon)):
        signed_area += polygon[i - 1, 0] * polygon[i, 1] - polygon[i, 0] * polygon[i - 1, 1]
    if signed_area > 0.:
        clockwise = True
    return clockwise


def order_faces_clockwise(faces, points, anticlockwise=False):
    new_faces = []
    for face in faces:
        if check_clockwise(points[face]) != anticlockwise:

            new_faces.append(face)
        else:
            new_faces.append(face[::-1])
    return new_faces


def outer_faces_from_faces(faces):
    quad_edge = faces_to_quad_edge(faces)
    edges = set([frozenset(edge) for edge, faces in quad_edge.items() if None in faces])
    edges = [list(edge) for edge in edges]
    return connect_edges(edges)


def mask_nodes(adjacency, mask):
    new_adjacency = {}
    for node, adjacents in adjacency.items():

        if mask[node]:
            new_adjacency[node] = [adjacent for adjacent in adjacents if mask[adjacent]]

    return new_adjacency


def polygon_area(points):
    return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))


def is_point_in_polygon(point, polygon):
    n = len(polygon)
    inside = False
    xints = 0.0
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if point[1] > min(p1y, p2y):
            if point[1] <= max(p1y, p2y):
                if point[0] <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or point[0] <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def forward_cyclical_distance(sequence, i, j, backward=False):
    i_idx = sequence.index(i)
    j_idx = sequence.index(j)

    if i_idx == j_idx:
        return 0

    if backward:
        if i_idx > j_idx:
            return i_idx - j_idx
        else:
            return len(sequence) + i_idx - j_idx
    else:
        if i_idx < j_idx:
            return j_idx - i_idx
        else:
            return len(sequence) - i_idx + j_idx


def remove_repetition_cyclical(sequence):
    new_sequence = []
    for i in range(len(sequence)):
        if sequence[i] != sequence[(i + 1) % len(sequence)]:
            new_sequence.append(sequence[i])

    counts = [sequence.count(i) for i in new_sequence]
    return new_sequence, counts


def count_clockwise_steps(adjacency, path, anticlockwise=False):
    steps = []
    for i in range(len(path) - 1):
        steps.append(forward_cyclical_distance(adjacency[path[i]], path[i - 1], path[i + 1], backward=anticlockwise))
    return steps


def faces_to_faces_surrounding_nodes(faces):
    faces_surrounding_nodes = defaultdict(list)
    for i, face in enumerate(faces):
        for node in face:
            faces_surrounding_nodes[node].append(i)
    return faces_surrounding_nodes


def faces_around_nodes(nodes, faces):
    faces_around_node = faces_to_faces_surrounding_nodes(faces)

    selection = set()
    for node in nodes:
        selection.update(faces_around_node[node])

    return list(selection)
