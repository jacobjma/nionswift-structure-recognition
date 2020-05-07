import cv2
import matplotlib
import numpy as np
from matplotlib import colors as mcolors

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
