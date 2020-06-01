import numpy as np


def traverse_left_most_outer(points, adjacency, counter_clockwise=True):
    left_most = np.where((points[:, 0] == np.min(points[:, 0])))[0]
    left_bottom_most = left_most[np.argmin(points[left_most, 1])]

    adjacent = adjacency[left_bottom_most]
    angles = np.arctan2(points[adjacent][:, 1] - points[left_bottom_most, 1],
                        points[adjacent][:, 0] - points[left_bottom_most, 0])

    if counter_clockwise:
        edge = (left_bottom_most, adjacent[np.argmin(angles)])
    else:
        edge = (left_bottom_most, adjacent[np.argmax(angles)])

    outer_path = [edge]
    while (edge != outer_path[0]) or (len(outer_path) == 1):
        next_adjacent = np.array(adjacency[edge[1]])
        j = next_adjacent[(np.nonzero(next_adjacent == edge[0])[0][0] - 1) % len(next_adjacent)]
        edge = (edge[1], j)
        outer_path.append(edge)

    return [edge[0] for edge in outer_path]


def count_clockwise_steps(path, adjacency):
    path = np.array(path)
    steps = []
    for i in range(len(path)):
        adjacent = adjacency[path[i]]
        n = np.nonzero(adjacent == path[i - 1])[0][0]
        m = np.nonzero(adjacent == path[(i + 1) % len(path)])[0][0]
        steps.append((m - n) % len(adjacent))
    return steps
