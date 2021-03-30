def bfs(adjacency, start):
    visited = [False] * (max(adjacency) + 1)

    queue = []
    queue.append(start)
    visited[start] = True
    order = []

    while queue:
        s = queue.pop(0)
        order.append(s)
        for i in adjacency[s]:
            if visited[i] == False:
                queue.append(i)
                visited[i] = True
    return order, visited


def connected_components(adjacency):
    to_start = set(adjacency.keys())
    components = []
    while len(to_start) > 0:
        n = to_start.pop()
        order, visited = bfs(adjacency, n)

        to_start = to_start - set(order)
        components.append(order)

    return components