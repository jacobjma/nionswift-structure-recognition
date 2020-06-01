import numpy as np


def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def labels_to_lists(labels, labelled):
    labelled = np.array(labelled)
    lists = []
    for i, indices in generate_indices(labels):
        lists.append(labelled[indices])
    return lists


def generate_indices(labels, first=0, last=None):
    if last is None:
        last = np.max(labels) + 1

    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(first, last)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield i, indices[l:h]


def subgraph_adjacency(node_indices, adjacency, relabel=True):
    if relabel:
        backward = {i: node_index for i, node_index in enumerate(node_indices)}
        forward = {value: key for key, value in backward.items()}
        node_indices = set(node_indices)
        adjacency = [set(adjacency[backward[i]]).intersection(node_indices) for i in range(len(node_indices))]
        return {i: [forward[adjacent] for adjacent in set(adjacency[backward[i]]).intersection(node_indices)] for i in
                range(len(node_indices))}
    else:
        node_indices = set(node_indices)
        return {i: list(set(adjacency[i]).intersection(node_indices)) for i in node_indices}


def connect_edges(edges):
    #print(edges)
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
    #print(connected_edges)
    while edges:
        connected_edges, edges = add_next_to_connected_edges(connected_edges, edges)
    #print(connected_edges)
    return connected_edges
