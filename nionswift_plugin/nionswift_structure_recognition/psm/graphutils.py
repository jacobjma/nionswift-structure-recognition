import numba as nb
import numpy as np

from psm.utils import ind2sub, check_clockwise
from psm.segments import RunLengthSegments
from collections import defaultdict


# TODO: fails with disconnected graph
@nb.njit
def linegraph_faces(points, indices, partitions, adjacency, adjacency_partitions):
    num_nodes = len(points)
    edge_centers = np.zeros((num_nodes * 3, 2), dtype=np.float64)

    marked = np.zeros(num_nodes, dtype=np.int32)
    linegraph_indices = np.full(num_nodes * 6 * 12, -1, dtype=np.int32)
    linegraph_partitions = np.full(num_nodes * 6, 0, dtype=np.int32)

    edge_dict = nb.typed.Dict.empty(
        key_type=nb.types.int32,
        value_type=nb.types.int32,
    )

    # inner nodes
    k = 0  # edge indexing
    l = 0  # face indexing
    for i in range(len(adjacency_partitions) - 1):
        adjacent = adjacency[adjacency_partitions[i]:adjacency_partitions[i + 1]]
        order = np.argsort(-np.arctan2(points[adjacent, 0] - points[i, 0],
                                       points[adjacent, 1] - points[i, 1]))
        adjacent = adjacent[order]
        if not np.any(i == indices[partitions[-2]:]):
            for j in range(len(adjacent)):
                if marked[adjacent[j]]:
                    linegraph_indices[linegraph_partitions[l] + j] = edge_dict[min(i, adjacent[j]) * num_nodes +
                                                                               max(i, adjacent[j])]
                else:
                    linegraph_indices[linegraph_partitions[l] + j] = k
                    edge_dict[min(i, adjacent[j]) * num_nodes + max(i, adjacent[j])] = k
                    edge_centers[k] = (points[i] + points[adjacent[j]]) / 2

                    marked[i] = 1
                    k += 1
            linegraph_partitions[l + 1] = linegraph_partitions[l] + len(adjacent)
            l += 1

    leftmost = np.argmin(points[:, 0])
    adjacent_points = points[adjacency[adjacency_partitions[leftmost]:adjacency_partitions[leftmost + 1]]] - points[
        leftmost]

    angles = np.arctan2(adjacent_points[:, 0], adjacent_points[:, 1])
    outer_leftmost_adjacent = adjacency[adjacency_partitions[leftmost]:adjacency_partitions[leftmost + 1]][
        np.argmax(angles)]

    edge = [leftmost, outer_leftmost_adjacent]
    outer = np.zeros(num_nodes * 30, dtype=np.int32)

    n = 0
    loop_start = 0
    j = edge[1]
    while j != leftmost:
        degree = adjacency_partitions[edge[0] + 1] - adjacency_partitions[edge[0]]
        adjacent = adjacency[adjacency_partitions[edge[0]]:adjacency_partitions[edge[0]] + degree]

        order = np.argsort(-np.arctan2(points[adjacent, 0] - points[edge[0], 0],
                                       points[adjacent, 1] - points[edge[0], 1]))
        adjacent = adjacent[order]

        insert = np.where(adjacent == edge[1])[0][0]

        for i in range(loop_start, degree):
            j = adjacent[(i + insert) % degree]

            if marked[j]:
                outer[n] = edge_dict[min(edge[0], j) * num_nodes + max(edge[0], j)]
                n += 1
            else:
                outer[n] = k
                n += 1
                edge_centers[k] = (points[edge[0]] + points[j]) / 2
                edge_dict[min(edge[0], j) * num_nodes + max(edge[0], j)] = k
                marked[edge[0]] = 1
                k += 1

        edge = [j, edge[0]]
        loop_start = 1

    edge_labels = np.zeros(num_nodes * 3, dtype=np.int32)

    # faces
    for i in range(len(partitions) - 2):
        face = indices[partitions[i]:partitions[i + 1]]
        for j in range(len(face)):
            edge_index = edge_dict[min(face[j - 1], face[j]) * num_nodes + max(face[j - 1], face[j])]
            linegraph_indices[linegraph_partitions[l] + j] = edge_index
            edge_labels[edge_index] += len(face)

        linegraph_partitions[l + 1] = linegraph_partitions[l] + len(face)
        l += 1

    linegraph_indices[linegraph_partitions[l]:linegraph_partitions[l] + n] = outer[:n]
    linegraph_partitions[l + 1] = linegraph_partitions[l] + n

    return (linegraph_indices[:linegraph_partitions[l] + n], linegraph_partitions[:l + 2], edge_centers[:k],
            edge_labels[:k])


@nb.njit
def dual_faces(points, indices, partitions, max_adjacent_faces=20):
    adjacent_faces = np.full((len(points), max_adjacent_faces), -1, dtype=np.int32)
    num_adjacent_faces = np.zeros(len(points), dtype=np.int32)

    # find indices of faces sharing a node, these are the dual node indices of the dual faces
    for i in range(len(partitions) - 1):
        face = indices[partitions[i]:partitions[i + 1]]
        for j in range(len(face)):
            adjacent_faces[face[j]][num_adjacent_faces[face[j]]] = i
            num_adjacent_faces[face[j]] += 1

    for i in range(len(adjacent_faces)):
        adjacent_edges = np.zeros(num_adjacent_faces[i], dtype=np.int32)

        for j in range(num_adjacent_faces[i]):
            face = indices[partitions[adjacent_faces[i][j]]:
                           partitions[adjacent_faces[i][j] + 1]]

            adjacent_edges[j] = face[np.where(face == i)[0][0] - 1]

        centered_points = points[adjacent_edges] - points[i]
        order = np.argsort(-np.arctan2(centered_points[:, 0], centered_points[:, 1]))
        adjacent_faces[i][:num_adjacent_faces[i]] = adjacent_faces[i][:num_adjacent_faces[i]][order]

    dual_indices = np.full(len(points) * max_adjacent_faces, -1, dtype=np.int32)
    dual_partitions = np.zeros(len(points), dtype=np.int32)

    outer = indices[partitions[-2]:]
    outer_edges = np.full((len(outer), 14), -1, dtype=np.int32)
    outer_piece_length = np.full(len(outer), -1, dtype=np.int32)

    k = 0
    l = 0
    for i in range(len(adjacent_faces)):

        in_outer = i == outer
        num_in_outer = np.sum(in_outer)
        if num_in_outer == 1:
            begin = np.where(adjacent_faces[i][:num_adjacent_faces[i]] == len(partitions) - 2)[0][0] + 1

            for j in range(num_adjacent_faces[i] - 2):
                outer_edges[np.where(i == outer)[0][0]][j] = adjacent_faces[i][(j + begin) % num_adjacent_faces[i]]
            outer_piece_length[np.where(i == outer)[0][0]] = num_adjacent_faces[i] - 2

        elif num_in_outer == 0:
            if num_adjacent_faces[i] > 2:  # remove two sided faces from dual
                dual_indices[l:l + num_adjacent_faces[i]] = adjacent_faces[i][:num_adjacent_faces[i]]
                dual_partitions[k] = l
                k += 1
                l += num_adjacent_faces[i]

        else:
            # TODO: support disconnected dual?
            raise RuntimeError('disconnected dual')

    dual_partitions[k] = l

    for i in range(len(outer_edges)):
        dual_indices[l:l + outer_piece_length[i]] = outer_edges[i][:outer_piece_length[i]]
        l += outer_piece_length[i]

    dual_partitions[k + 1] = l
    dual_partitions = dual_partitions[:k + 2]
    dual_indices = dual_indices[:l]

    return dual_indices, dual_partitions


def adjacency_to_faces(points, indices, partitions):
    num_nodes = len(partitions) - 1
    edge_set = set()
    faces_indices = np.zeros(num_nodes * 10, dtype=np.int32)
    faces_partitions = np.zeros(num_nodes * 3, dtype=np.int32)

    outer_face = traverse_outer(points, indices, partitions)

    for i in range(len(outer_face)):
        edge_set.add(outer_face[i] + outer_face[(i + 1) % len(outer_face)] * num_nodes)

    m = 0
    n = 1
    for i in range(num_nodes):

        adjacent = indices[partitions[i]:partitions[i + 1]]

        for j in adjacent:
            k = i
            if not (i * num_nodes + j) in edge_set:
                faces_indices[m] = k
                m += 1
                edge_set.add(i * num_nodes + j)

                while j != i:
                    next_adjacent = indices[partitions[j]:partitions[j + 1]]
                    l = (np.nonzero(next_adjacent == k)[0][0] - 1) % len(next_adjacent)
                    k = j
                    j = next_adjacent[l]
                    faces_indices[m] = k
                    m += 1
                    edge_set.add(k * num_nodes + j)

                faces_partitions[n] = m
                n += 1

    faces_indices[m:m + len(outer_face)] = outer_face[::-1]
    faces_partitions[n] = m + len(outer_face)
    return faces_indices[:m + len(outer_face)], faces_partitions[:n + 1]


def traverse_outer(points, indices, partitions):
    left_most = np.where((points[:, 0] == np.min(points[:, 0])))[0]
    left_bottom_most = left_most[np.argmin(points[left_most, 1])]

    adjacent = indices[partitions[left_bottom_most]:partitions[left_bottom_most + 1]]

    angles = np.arctan2(points[adjacent][:, 1] - points[left_bottom_most, 1],
                        points[adjacent][:, 0] - points[left_bottom_most, 0])

    other = adjacent[np.argmin(angles)]
    outer_indices = np.zeros(len(partitions) * 2, dtype=np.int32)
    outer_indices[0] = other

    i = left_bottom_most
    j = other
    m = 1
    while j != left_bottom_most:
        next_adjacent = indices[partitions[j]:partitions[j + 1]]
        l = (np.nonzero(next_adjacent == i)[0][0] + 1) % len(next_adjacent)
        i = j
        j = next_adjacent[l]
        outer_indices[m] = j
        m += 1

    return outer_indices[:m]


@nb.njit
def outer_face_from_inner(indices, partitions, num_nodes):
    edge_set = set()

    for i in range(len(partitions) - 1):
        face = indices[partitions[i]:partitions[i + 1]]
        for j in range(len(face)):
            edge = min(face[j - 1], face[j]) * num_nodes + max(face[j - 1], face[j])

            if edge in edge_set:
                edge_set.remove(edge)

            else:
                edge_set.add(edge)

    edges = np.zeros((num_nodes * 6, 2), dtype=np.int32)

    i = 0
    for edge in edge_set:
        edges[i] = ind2sub(num_nodes, edge)
        i += 1

    return order_looping_edges(edges[:i])


def remove_outer_adjacent(faces, points):
    indices, partitions = _remove_outer_adjacent(faces.indices, faces.partitions, points)
    return RunLengthSegments(indices, partitions)


def _remove_outer_adjacent(indices, partitions, points):
    outer = indices[partitions[-2]:]

    new_indices = np.zeros(indices.shape, dtype=indices.dtype)
    new_partitions = np.zeros(partitions.shape, dtype=partitions.dtype)

    j = 0
    k = 1
    for i in range(len(partitions) - 1):
        face = indices[partitions[i]:partitions[i + 1]]
        if set(face).isdisjoint(outer):
            new_indices[j:j + len(face)] = face
            new_partitions[k] = j + len(face)
            j += len(face)
            k += 1

    new_indices = new_indices[:j]
    new_partitions = new_partitions[:k]

    outer = outer_face_from_inner(new_indices, new_partitions, len(points))[:, 0]

    if check_clockwise(points[outer]):
        outer = outer[::-1]

    new_indices = np.append(new_indices, outer)
    new_partitions = np.append(new_partitions, len(new_indices))

    return new_indices, new_partitions


def remove_nodes(indices, partitions, to_remove):
    remapping = np.full(len(partitions) - 1, -1, dtype=np.int32)
    j = 0
    for i in range(len(partitions) - 1):
        if i in to_remove:
            j += 1
        else:
            remapping[i] = i - j

    new_indices = np.zeros(indices.shape, dtype=indices.dtype)
    new_partitions = np.zeros(partitions.shape, dtype=partitions.dtype)

    k = 0
    l = 0
    for i in range(len(partitions) - 1):
        if remapping[i] > -1:
            for j in indices[partitions[i]:partitions[i + 1]]:
                if remapping[j] > -1:
                    new_indices[k] = remapping[j]
                    k += 1
            l += 1
            new_partitions[l] = k

    return new_indices[:k], new_partitions[:l + 1]


@nb.njit
def faces_to_adjacency(indices, partitions, num_nodes, max_degree=100, include_self=False):
    adjacency = np.full((num_nodes, max_degree), -1, dtype=np.int32)
    degrees = np.zeros(num_nodes, dtype=np.int32)

    if include_self:
        adjacency[:, 0] = np.arange(0, len(adjacency)).astype(np.int32)
        degrees += 1

    for i in range(len(partitions) - 1):
        face = indices[partitions[i]:partitions[i + 1]]
        for j in range(len(face)):
            adjacency[face[j]][degrees[face[j]]] = face[j - 1]
            degrees[face[j]] += 1

    partitions = np.zeros(len(adjacency) + 1, dtype=np.int32)
    indices = np.zeros(np.sum(degrees), dtype=np.int32)
    for i in range(len(adjacency)):
        partitions[i + 1] = partitions[i] + degrees[i]
        indices[partitions[i]:partitions[i + 1]] = adjacency[i][:degrees[i]]

    return indices, partitions


@nb.njit
def adjacency_to_edges(indices, partitions):
    num_edges = np.sum(np.diff(partitions))
    edges = np.zeros((num_edges, 2), dtype=np.int32)

    k = 0
    for i in range(len(partitions) - 1):
        adjacent = indices[partitions[i]:partitions[i + 1]]
        for j in adjacent:
            if i < j:
                edges[k][0] = i
                edges[k][1] = j
                k += 1

    return edges[:k]


@nb.njit
def order_looping_edges(edges):
    N = len(edges)
    result = []
    k = 0
    while k < N - 1:
        ordered = np.zeros(edges.shape, edges.dtype)
        marked = np.zeros(len(edges), dtype=np.int32)

        ordered[0] = edges[0]
        marked[0] = 1
        i = 0

        while i < len(edges) - 1:
            f = False
            for j in range(len(edges)):
                edge = edges[j]
                if marked[j] == 0:
                    if edge[0] == ordered[i][1]:
                        i += 1
                        k += 1
                        ordered[i] = edge
                        marked[j] = 1
                        f = True
                        break

                    elif edge[1] == ordered[i][1]:
                        i += 1
                        k += 1
                        ordered[i] = edge[::-1]
                        marked[j] = 1
                        f = True
                        break
            if f == False:
                break

        result.append(ordered[:i + 1])
        edges = edges[marked == 0]

        if len(edges) == 0:
            break

    return result


def edge_adjacent_faces(faces, points):
    num_nodes = len(points)

    indices = np.zeros(len(faces.indices) * faces.sizes().max(), dtype=np.int32)
    partitions = np.zeros(len(faces.partitions) * faces.sizes().max(), dtype=np.int32)

    quadedge = defaultdict(list)
    for i, face in enumerate(faces):
        for j in range(len(face)):
            k = (j + 1) % len(face)
            quadedge[frozenset((face[j], face[k]))].append(i)

    new_quadedge = {}
    m = 1
    for edge_faces in quadedge.values():
        if not (np.array(edge_faces) == len(faces) - 1).any():
            face_1 = faces[edge_faces[0]]
            face_2 = faces[edge_faces[1]]
            i = np.concatenate((face_1, face_2))
            p = np.array([0, len(face_1), len(face_1) + len(face_2)])

            try:
                loop = outer_face_from_inner(i, p, num_nodes)[:, 0]

                if not check_clockwise(points[loop]):
                    loop = loop[::-1]

                new_quadedge[m - 1] = edge_faces
                indices[k:k + len(loop)] = loop
                k += len(loop)
                partitions[m] = k
                m += 1
            except:
                pass

    return RunLengthSegments(indices[:k], partitions[:m]), new_quadedge
