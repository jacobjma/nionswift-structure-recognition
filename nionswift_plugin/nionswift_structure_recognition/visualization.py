import cv2
import matplotlib
import numpy as np
from matplotlib import colors as mcolors

from .widgets import StructureRecognitionModule
from .widgets import Section, combo_box_template, check_box_template, line_edit_template

named_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


def get_colors_from_cmap(c, cmap=None, vmin=None, vmax=None):
    if cmap is None:
        cmap = matplotlib.cm.get_cmap('viridis')

    elif isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    if vmin is None:
        vmin = np.nanmin(c)

    if vmax is None:
        vmax = np.nanmax(c)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    c = np.array(c, dtype=float)

    valid = np.isnan(c) == 0
    colors = np.zeros((len(c), 4))
    colors[valid] = cmap(norm(c[valid]))

    return colors


def add_faces(points, faces, image, colors):
    points = np.round(points).astype(int)

    for face, color in zip(faces, colors):
        cv2.fillConvexPoly(image, points[face][:, ::-1], tuple(map(int, color)))

    return image


def add_edges(points, edges, image, color, thickness=1):
    points = np.round(points).astype(int)
    for edge in edges:
        cv2.line(image, tuple(points[edge[0]]), tuple(points[edge[1]][::-1]), color=color,
                 thickness=thickness)

    return image


def add_points(points, image, size, colors):
    points = np.round(points).astype(np.int)

    for point, color in zip(points, colors):
        cv2.circle(image, (point[0], point[1]), size, tuple(map(int, color)), -1)

    return image


# class GraphicOptions(object):
#
#     def __init__(self, graphic_check_box, contents_column):
#         self.graphic_check_box = graphic_check_box
#         self.column = ui.cre
#         self.contents_column = contents_column
#
#     def check_box_changed(self):
#         self.contents_column._widget.remove_all()



    def create_background(self, image, density, segmentation):

        if self.background == 'image':
            visualization = ((image - image.min()) / image.ptp() * 255).astype(np.uint8)
            visualization = np.tile(visualization[..., None], (1, 1, 3))

        elif self.background == 'density':
            visualization = (density * 255).astype(np.uint8)
            visualization = np.tile(visualization[..., None], (1, 1, 3))

        elif self.background == 'segmentation':
            visualization = (segmentation / segmentation.max() * 255).astype(np.uint8)
            visualization = np.tile(visualization[..., None], (1, 1, 3))

        elif self.background == 'solid':
            visualization = None

        else:
            raise RuntimeError()

        return visualization

    def add_points(self, visualization, points):
        if self.overlay_points:
            if self.points_color == 'solid':
                color = mcolors.to_rgba(named_colors[self.points_color_solid])[:3]
                colors = [tuple([int(x * 255) for x in color[::-1]])] * len(points)

            # elif self.points_color == 'class':
            #     colors = (get_colors_from_cmap(probabilities[:, 2], 'autumn', vmin=0, vmax=1) * 255).astype(int)
            #     colors = colors[:, :-1][:, ::-1]

            else:
                raise NotImplementedError()

            visualization = add_points(points, visualization, self.points_size, colors)

        return visualization

    def add_edges(self, visualization):
        pass
        #return add_edges(graph.points, graph.edges(), visualization, (0, 0, 0), self.line_width)

    def add_faces(self, visualization):
        pass
        # if self.overlay_faces:
        #     if self.faces_color == 'size':
        #         colors = graph.faces().sizes()
        #         vmin = 0
        #         vmax = 10
        #
        #     elif self.faces_color == 'rmsd':
        #         colors = rmsd
        #         vmin = 0
        #         vmax = np.max(rmsd[rmsd != np.inf])
        #
        #     else:
        #         raise RuntimeError()
        #
        #     colors = (get_colors_from_cmap(colors, self.faces_cmap, vmin, vmax) * 255).astype(int)
        #
        #     visualization = add_faces(graph.points, graph.faces()[:-1], visualization, colors)
