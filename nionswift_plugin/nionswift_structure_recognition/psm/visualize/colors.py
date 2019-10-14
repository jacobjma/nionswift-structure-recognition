import matplotlib
import numpy as np


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


def assign_colors(labels, color_assigment):
    colors = np.zeros((len(labels), 4), dtype=float)
    for i, label in enumerate(labels):
        #print(label)
        try:
            #print(label, colors[i])
            colors[i] = matplotlib.colors.to_rgba(color_assigment[label])
            #print(label,colors[i])
        except KeyError:
            colors[i] = matplotlib.colors.to_rgba(color_assigment[-1])

    return colors