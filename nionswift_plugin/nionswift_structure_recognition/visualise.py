import cv2
import numpy as np
from matplotlib import colors
import matplotlib
import matplotlib.cm


def convert2color(image):
    image = (image - image.min()) / (image.max() - image.min())
    return np.tile(np.uint8(255 * image), (1, 1, 1, 3))

def add_points(image, positions, colors=None, size=2):
    positions_rounded = np.round(positions).astype(np.int)

    if colors is None:
        colors = [(0, 0, 255)] * len(positions)

    for position, color in zip(positions_rounded, colors):
        # cv2.circle(image, tuple(center), size, colors[class_id], 2)
        cv2.circle(image, (position[1], position[0]), size, color, -1)

    return image


def add_line_segments(image, line_segments):
    for line_segment in np.round(line_segments).astype(np.int):
        cv2.line(image, (line_segment[0][1], line_segment[0][0]),
                 (line_segment[1][1], line_segment[1][0]), color=(0, 0, 0),
                 thickness=2)

    return image


def add_polygons(image, polygons, colors):
    for polygon, color in zip(polygons, colors):
        outline = np.round(polygon).astype(int)

        cv2.fillConvexPoly(image, outline, color)

    return image


def get_colors_array(c, n=None, vmin=None, vmax=None, cmap=None):
    try:
        color = matplotlib.colors.to_rgba_array(c)
    except:

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
    else:
        colors = np.zeros((n, 4))
        colors[:] = matplotlib.colors.to_rgba_array(color)

    return colors
