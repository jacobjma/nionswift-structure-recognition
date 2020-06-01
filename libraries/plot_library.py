import matplotlib.pyplot as plt
import numpy as np

from psm.libraries import load_library
from psm.structures.graphene import defect_fingerprint

lib, alias = load_library('graphene')

ncols = 3
nrows = int(np.ceil(len(lib) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
axes = axes.ravel()
for ax, graph in zip(axes, lib.values()):
    short_signature = defect_fingerprint(graph, True)

    try:
        ax.set_title(short_signature + '\n' + alias[short_signature])
    except:
        ax.set_title(short_signature + '\n' + 'No alias')

    graph.plot(ax=ax, point_colors={0: 'k', 1: 'lime', }, point_kwargs={'s': 100},
               line_kwargs={'linewidth': 2, 'colors': 'k'})
    ax.axis('equal')
    ax.axis('off')

for ax in axes[len(lib):]:
    fig.delaxes(ax)

plt.tight_layout()
plt.savefig('graphene.pdf')