from collections import defaultdict

from .utils import flatten_list_of_lists


def faces_to_faces_surrounding_nodes(faces):
    faces_surrounding_nodes = defaultdict(list)
    for i, face in enumerate(faces):
        for node in face:
            faces_surrounding_nodes[node].append(i)
    return faces_surrounding_nodes


def select_faces_around_nodes(nodes, faces):
    faces_around_node = faces_to_faces_surrounding_nodes(faces)

    selection = set()
    for node in nodes:
        selection.update(faces_around_node[node])

    return list(selection)


def select_surrounded_faces(nodes, faces):
    mask = set(nodes)
    selected = []
    for i, face in enumerate(faces):
        if set(face).issubset(mask):
            selected.append(i)
    return selected


def select_nodes_in_faces(selected_faces, faces):
    return list(set(flatten_list_of_lists([faces[node] for node in selected_faces])))


def grow(indices, adjacency, remove_initial=False):
    new_indices = [item for i in indices for item in adjacency[i]] + list(indices)
    if remove_initial:
        return list(set(new_indices) - set(indices))
    else:
        return list(set(new_indices))
