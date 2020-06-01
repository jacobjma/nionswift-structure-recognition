from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from .utils import connect_edges


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
            edges.add(frozenset({face[i - 1], face[i]}))

    return [list(edge) for edge in edges]


def faces_to_quad_edge(faces):
    quad_edge = defaultdict(list)
    for i, face in enumerate(faces):
        #if 226 in face:
        #    print(face)
        for j in range(len(face)):
            quad_edge[frozenset({face[j - 1], face[j]})].append(i)
    return quad_edge


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


def edges_to_matrix(edges, num_nodes=None):
    row_ind = [edge[0] for edge in edges]
    col_ind = [edge[1] for edge in edges]

    if num_nodes is None:
        num_nodes = max(max(row_ind), max(col_ind))

    return csr_matrix((np.ones(len(edges), dtype=np.bool), (row_ind, col_ind)), (num_nodes,) * 2, dtype=np.bool)


def adjacency_to_matrix(adjacency, num_nodes=None):
    if num_nodes is None:
        num_nodes = max(adjacency.keys()) + 1

    edges = adjacency_to_edges(adjacency)
    return edges_to_matrix(edges, num_nodes)


def faces_to_node_connected_faces_adjacency(faces):
    faces_surrounding_nodes = faces_to_faces_surrounding_nodes(faces)

    adjacency = defaultdict(set)
    for faces in faces_surrounding_nodes.values():
        for face in faces:
            adjacency[face].update(set(faces) - {face})

    return adjacency


def outer_faces_from_faces(faces):
    quad_edge = faces_to_quad_edge(faces)
    edges = [list(edge) for edge, faces in quad_edge.items() if len(faces) == 1]
    return connect_edges(edges)
