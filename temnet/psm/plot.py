import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Polygon


def add_edges_to_mpl_plot(points, edges, ax=None, **kwargs):
    if ax is None:
        ax = plt.subplot()
    line_collection = LineCollection(points[edges], **kwargs)
    ax.add_collection(line_collection)
    ax.autoscale()
    return line_collection


def add_polygons_to_mpl_plot(polygons, facecolors=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.subplot()

    patches = []
    for polygon in polygons:
        patches.append(Polygon(polygon, closed=True))

    patch_collection = PatchCollection(patches, facecolors=facecolors, **kwargs)
    ax.add_collection(patch_collection)
    ax.autoscale()
    return patch_collection


