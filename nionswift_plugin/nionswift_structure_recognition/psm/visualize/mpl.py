import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_edges(ax, points, edges, colors='k', linewidths=1.):
    line_collection = LineCollection(points[edges], colors=colors, linewidths=linewidths)
    ax.add_collection(line_collection)
    ax.autoscale()
    return line_collection


def add_polygons(ax, points, polygons, colors):
    patches = []
    for polygon in polygons:
        patches.append(Polygon(points[polygon], closed=True))
    #print(colors.min())
    patch_collection = PatchCollection(patches, facecolors=colors)
    ax.add_collection(patch_collection)
    return patch_collection


def add_dual_color_edges(ax, points, edges, c, cmap=None, n=2, vmin=None, vmax=None, **kwargs):
    if cmap is None:
        cmap = matplotlib.cm.get_cmap('viridis')

    elif isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    colors = np.zeros((n * len(edges), 4))
    lines = np.zeros((n * len(edges), 2, 2))
    for i, edge in enumerate(edges):

        edge = list(edge)

        p1, p2 = points[edge[0]], points[edge[1]]
        c1, c2 = norm(c[edge[0]]), norm(c[edge[1]])

        if np.isnan(c1) & np.isnan(c2):
            colors[i * n:i * n + n] = np.zeros((n, 4))

        elif np.isnan(c1):
            colors[i * n:i * n + n] = cmap(c2)
            colors[i * n:i * n + n, 3] = np.linspace(0, 1, n)

        elif np.isnan(c2):
            colors[i * n:i * n + n] = cmap(c1)
            colors[i * n:i * n + n, 3] = np.linspace(0, 1, n)[::-1]

        else:
            colors[i * n:i * n + n] = cmap(c1 + (c2 - c1) * np.linspace(0, 1, n))

        line = p1[None, :] + (p2 - p1)[None, :] * np.linspace(0, 1, n + 1)[:, None]
        line = line.reshape(-1, 1, 2)
        lines[i * n:i * n + n] = np.concatenate([line[:-1], line[1:]], axis=1)

    lc = LineCollection(lines, colors=colors, cmap=cmap, **kwargs)
    ax.add_collection(lc)
    ax.autoscale()

    # lc.set_array(np.array(c))
    # lc.set_clim([vmin, vmax])

    return lc, ax


def add_colorbar(ax, cmap, vmin, vmax, n_ticks=5, ticks=None, orientation='horizontal', position='bottom', size='5%',
                 pad=0.05, **kwargs):
    if position in ['right', 'left']:
        orientation = 'vertical'
    elif position in ['top', 'bottom']:
        orientation = 'horizontal'

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    if ticks is None:
        ticks = np.linspace(vmin, vmax, n_ticks)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    return plt.colorbar(sm, cax=cax, orientation=orientation, ticks=ticks, **kwargs)
