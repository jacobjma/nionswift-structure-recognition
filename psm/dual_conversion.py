import numpy as np
from scipy.sparse import csr_matrix

from .representation import faces_to_quad_edge, edges_to_adjacency, faces_to_adjacency
from .utils import connect_edges


def faces_to_dual_edges(faces):
    return [edge for edge in faces_to_quad_edge(faces).values() if len(edge) == 2]


def faces_to_dual_adjacency(faces):
    return edges_to_adjacency(faces_to_dual_edges(faces))


def faces_to_dual_matrix(faces):
    dual_edges = faces_to_dual_edges(faces)
    row_ind = [edge[0] for edge in dual_edges]
    col_ind = [edge[1] for edge in dual_edges]
    return csr_matrix((np.ones(len(dual_edges), dtype=np.bool), (row_ind, col_ind)), (len(faces),) * 2, dtype=np.bool)


def faces_to_dual_faces(faces, num_nodes):
    quad_edge = faces_to_quad_edge(faces)
    adjacency = faces_to_adjacency(faces, num_nodes)

    dual_faces = []
    for central_node, adjacent_nodes in adjacency.items():
        new_dual_faces = []

        if len(adjacent_nodes) == 0:
            continue
        is_face = True

        for adjacent_node in adjacent_nodes:
            new_dual_faces.append(quad_edge[frozenset((central_node, adjacent_node))])
            if len(new_dual_faces[-1]) == 1:
                is_face = False
                break

        #print(new_dual_faces, is_face)
        if is_face & (len(new_dual_faces) > 2):

            dual_faces.append(new_dual_faces)

    return [connect_edges(edges)[0] for edges in dual_faces]
