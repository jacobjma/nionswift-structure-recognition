import matplotlib
import numpy as np
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Polygon, Rectangle


def add_edges_as_line_collection(ax, points, edges, **kwargs):
    line_collection = LineCollection(points[edges], **kwargs)
    ax.add_collection(line_collection)
    ax.autoscale()
    return line_collection


def add_polygons(ax, polygons, colors):
    if isinstance(colors, dict):
        colors = [colors[len(polygon)] for polygon in polygons]
    patches = []
    for polygon in polygons:
        patches.append(Polygon(polygon, closed=True))

    patch_collection = PatchCollection(patches, facecolors=colors)
    ax.add_collection(patch_collection)
    return patch_collection


def add_rectangles(ax, rectangles):
    patches = []
    for rectangle in rectangles:
        patches.append(Rectangle(xy=(rectangle[0, 0], rectangle[1, 0]),
                                 width=rectangle[0, 1] - rectangle[0, 0],
                                 height=rectangle[1, 1] - rectangle[1, 0]
                                 ))
    patch_collection = PatchCollection(patches, facecolors='none', edgecolors='r')
    ax.add_collection(patch_collection)
    return patch_collection


def assign_colors(labels, color_assigment):
    colors = np.zeros((len(labels), 4), dtype=float)
    for i, label in enumerate(labels):
        try:
            colors[i] = matplotlib.colors.to_rgba(color_assigment[label])
        except KeyError:
            colors[i] = matplotlib.colors.to_rgba(color_assigment[-1])

    return colors
