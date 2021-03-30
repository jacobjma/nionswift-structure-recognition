import bisect
import numpy as np


def prioritized_greedy_two_coloring(adjacency, priority):
    labels = np.full(len(adjacency), -1, dtype=np.int)
    first_node = np.argmin(priority)

    queue = [(priority[first_node], first_node)]

    while queue:
        _, node = queue.pop(0)
        neighbors = np.array(adjacency[node])
        neighbors = neighbors[labels[neighbors] == -1]
        labels[neighbors] = labels[node] == 0

        for neighbor in neighbors:
            bisect.insort(queue, (priority[neighbor], neighbor))

    return labels
