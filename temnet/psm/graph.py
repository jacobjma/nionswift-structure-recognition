import matplotlib.pyplot as plt
import numpy as np

from temnet.psm.dual import faces_to_dual_faces
from temnet.psm.plot import add_edges_to_mpl_plot
from temnet.psm.stable_delaunay_graph import stable_delaunay_faces
from temnet.psm.utils import faces_to_adjacency, faces_to_edges, order_adjacency_clockwise


def stable_delaunay_graph(points, alpha):
    return GeometricGraph(points, stable_delaunay_faces(points, alpha))


class GeometricGraph:

    def __init__(self, points, faces):
        self._points = points
        self._faces = faces
        self._adjacency = None
        self._edges = None

    @property
    def points(self):
        return self._points

    # def calculate_faces(self):
    #     if len(self.points) == 0:
    #         return [[]]
    #     return

    def calculate_adjacency(self):
        adjacency = faces_to_adjacency(self.faces, len(self.points))
        return order_adjacency_clockwise(self.points, adjacency)

    def calculate_edges(self):
        return faces_to_edges(self.faces)

    def _cached_property(self, name, calculator):
        if getattr(self, name) is not None:
            return getattr(self, name)
        else:
            setattr(self, name, calculator())
            return getattr(self, name)

    @property
    def dual(self):
        dual_faces = faces_to_dual_faces(self.faces, len(self.points))
        return self.__class__(self.face_centers, dual_faces)

    @property
    def faces(self):
        return self._faces  # self._cached_property('_faces', self.calculate_faces)

    @property
    def face_centers(self):
        return np.array([self.points[face].mean(0) for face in self.faces])

    @property
    def adjacency(self):
        return self._cached_property('_adjacency', self.calculate_adjacency)

    @property
    def edges(self):
        return self._cached_property('_edges', self.calculate_edges)

    def show(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        add_edges_to_mpl_plot(self.points, edges=self.edges, ax=ax)

        # for i, face_center in enumerate(graph.face_centers):
        #     ax.annotate(f'{i}', xy=face_center)
        #
        # for i, p in enumerate(graph.points):
        #     ax.annotate(f'{i}', xy=p)
