import copy
import json
import os
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components

from .construct import stable_delaunay_faces
from .dual_conversion import faces_to_dual_faces
from .representation import faces_to_edges, faces_to_adjacency, outer_faces_from_faces, adjacency_to_matrix, \
    order_adjacency_clockwise
from .select import grow
from .traverse import traverse_left_most_outer
from .utils import flatten_list_of_lists, labels_to_lists
from .visualize import add_edges_as_line_collection, add_polygons


def stable_delaunay_graph(points, alpha, r=np.inf):
    faces = stable_delaunay_faces(points, alpha, r)
    return GeometricGraph(points, faces)


class GeometricGraphBase:

    @property
    def points(self):
        raise NotImplementedError()

    @property
    def labels(self):
        raise NotImplementedError()

    @property
    def adjacency(self):
        raise NotImplementedError()

    @property
    def edges(self):
        raise NotImplementedError()

    @property
    def faces(self):
        raise NotImplementedError()

    @property
    def degrees(self):
        return [len(adjacent) for adjacent in self.adjacency.values()]

    @property
    def face_polygons(self):
        return [self.points[face] for face in self.faces]

    @property
    def face_labels(self):
        return [self.labels[face] for face in self.faces]

    def plot(self, ax=None, point_colors=None, face_colors=None, point_kwargs=None, line_kwargs=None):
        if ax is None:
            ax = plt.subplot()

        if point_kwargs is None:
            point_kwargs = {}

        if line_kwargs is None:
            line_kwargs = {}

        if face_colors is not None:
            add_polygons(ax, self.face_polygons, face_colors)

        add_edges_as_line_collection(ax, self.points, self.edges, **line_kwargs)

        if point_colors is not None:
            point_colors = [point_colors[label] for label in self.labels]
            ax.scatter(*self.points.T, c=point_colors, zorder=2, **point_kwargs)


class GeometricGraph(GeometricGraphBase):

    def __init__(self, points, faces, labels=None):
        self._points = np.array(points)
        self._faces = faces

        if labels is not None:
            assert len(labels) == len(points)

        self.set_labels(labels)

    def __len__(self):
        return len(self.points)

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, values):
        self.points[:] = values

    @property
    def faces(self):
        return self._faces

    @property
    def labels(self):
        return self._labels

    def set_labels(self, labels):
        if labels is None:
            self._labels = None
            return

        if len(labels) != len(self):
            raise RuntimeError('Number of labels does not match size of graph')

        labels = np.array(labels)

        if not np.issubdtype(labels.dtype, np.integer):
            raise RuntimeError('Labels must be integers')

        self._labels = labels

    @property
    @lru_cache(maxsize=1)
    def adjacency(self):
        return order_adjacency_clockwise(self.points, faces_to_adjacency(self.faces, len(self)))

    @property
    @lru_cache(maxsize=1)
    def edges(self):
        return np.array(faces_to_edges(self.faces))

    def append(self, other):
        other_faces = [[node + len(self) for node in face] for face in other.faces]
        self._points = np.vstack((self.points, other.points))
        self._faces = self.faces + other_faces
        self._labels = np.concatenate((self.labels, other.labels))

    def outer_faces(self):
        return outer_faces_from_faces(self.faces)

    def outer_face_polygons(self):
        return [self.points[face] for face in self.outer_faces()]

    def dual(self):
        dual_points = np.zeros((len(self.faces), 2))
        for i, face in enumerate(self.faces):
            dual_points[i] = self.points[face].mean(axis=0)
        dual_faces = faces_to_dual_faces(self._faces, len(self))
        return self.__class__(dual_points, dual_faces)

    def subgraph_from_faces(self, indices):
        if len(indices) == 0:
            raise RuntimeError('Empty list of indices')
        return SubgraphFromFaces(indices, self)

    def subgraph_from_nodes(self, indices):
        if len(indices) == 0:
            raise RuntimeError('Empty list of indices')
        return SubgraphFromNodes(indices, self)

    def delete_faces(self, faces_to_delete, return_ordering=False):
        faces_to_keep = np.delete(np.arange(len(self.faces)), faces_to_delete)
        subgraph = self.subgraph_from_faces(faces_to_keep)
        if return_ordering:
            return subgraph.detach(), subgraph.member_order
        else:
            return subgraph.detach()

    def write(self, path, overwrite=False):
        _, ext = os.path.splitext(path)

        if ext != '.json':
            path = path + '.json'

        if os.path.isfile(path) & (not overwrite):
            raise RuntimeError('file {} already exists'.format(path))

        with open(path, 'w') as f:
            dict_to_json = {'points': self.points.tolist(), 'faces': self.faces}

            if self.labels is not None:
                dict_to_json.update({'labels': self.labels.tolist()})

            f.write(json.dumps(dict_to_json))

    @classmethod
    def read(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
            graph = cls(data['points'], data['faces'])
            if 'labels' in data.keys():
                graph.set_labels(data['labels'])
        return graph

    def copy(self):
        new_copy = self.__class__(self.points.copy(), copy.deepcopy(self.faces))
        if self.labels is not None:
            new_copy.set_labels(self.labels.copy())
        return new_copy


class Subgraph(GeometricGraphBase):

    def __init__(self, parent):
        self._parent = parent

    def __len__(self):
        return len(self.member_nodes)

    @property
    def labels(self):
        if self.parent.labels is None:
            return
        return self.parent.labels[self.member_nodes]

    @property
    def member_nodes(self):
        raise NotImplementedError()

    @property
    def points(self):
        return self.parent.points[self.member_nodes]

    @property
    @lru_cache(maxsize=1)
    def member_order(self):
        return {member_node: i for i, member_node in enumerate(self.member_nodes)}

    @property
    @lru_cache(maxsize=1)
    def reverse_member_order(self):
        return {value: key for key, value in self.member_order.items()}

    @property
    def member_faces(self):
        raise NotImplementedError()

    @property
    def parent(self):
        return self._parent

    @property
    @lru_cache(maxsize=1)
    def adjacency(self):
        parent_adjacency = self.parent.adjacency
        adjacency = [set(parent_adjacency[n]).intersection(self.member_nodes) for n in self.member_nodes]
        adjacency = {i: [self.member_order[node] for node in nodes] for i, nodes in enumerate(adjacency)}
        return order_adjacency_clockwise(self.points, adjacency)

    @property
    def matrix(self):
        return adjacency_to_matrix(self.adjacency, num_nodes=len(self))

    def detach(self):
        raise NotImplementedError()


class SubgraphFromFaces(Subgraph):

    def __init__(self, member_faces, parent):
        self._member_faces = member_faces
        self._parent = parent
        super().__init__(parent)

    @property
    def member_faces(self):
        return self._member_faces

    @property
    def member_nodes(self):
        return list(set(flatten_list_of_lists([self.parent.faces[i] for i in self.member_faces])))

    @property
    def faces(self):
        faces = [self.parent.faces[member_face] for member_face in self.member_faces]
        return [[self.member_order[node] for node in face] for face in faces]

    def detach(self):
        return self.parent.__class__(np.array(self.points), self.faces, self.labels)


class SubgraphFromNodes(Subgraph):

    def __init__(self, member_nodes, parent):
        self._member_nodes = member_nodes
        super().__init__(parent)

    @property
    def member_nodes(self):
        return self._member_nodes

    def grow(self, remove_initial=False):
        return self.__class__(grow(self.member_nodes, self.parent.adjacency, remove_initial=remove_initial),
                              self.parent,
                              )

    def connected_components(self):
        labels = connected_components(self.matrix)[1]
        return [self.__class__(nodes, self.parent) for nodes in labels_to_lists(labels, self.member_nodes)]

    def enclosing_path(self):
        ordered_adjacency = order_adjacency_clockwise(self.points, self.adjacency)
        path = traverse_left_most_outer(self.points, ordered_adjacency)
        return path[:-1]
