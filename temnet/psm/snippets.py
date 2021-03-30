import scipy
from temnet.psm.plot import get_colors_from_cmap

delaunay = scipy.spatial.Delaunay(points)
simplices = delaunay.simplices

edges = faces_to_edges(simplices)
adjacency = order_adjacency_clockwise(points, faces_to_adjacency(simplices, len(points)))

edge_rmsd = {frozenset(edge):np.inf for edge in edges}
for i in range(len(points)):
    adjacent = adjacency[i]

    adjacent_combinations = list(combinations(adjacent, 3))
    src = points[np.array([(i,) + tuple(combination) for combination in adjacent_combinations])] - points[i][None,None]

    rmsds = batch_rmsd_qcp(src, template[None])

    for rmsd, combination in zip(rmsds, adjacent_combinations):
        for j in combination:
            edge_rmsd[frozenset((i, j))] = min(rmsd, edge_rmsd[frozenset((i, j))])


j = 2

edges = [edge for edge in edges if edge_rmsd[frozenset(edge)] < 5]

colors = [edge_rmsd[frozenset(edge)] for edge in edges]
colors = get_colors_from_cmap(colors)
#print(colors)
#plt.plot(*adjacent_comb[j].T, 'ro')
#plt.axis('equal')
#plt.plot(*template.T)

fig,ax=plt.subplots(figsize=(14,14))

add_edges_to_mpl_plot(points, edges,ax=ax,colors=colors)
ax.plot(*points.T,'ko')

#for i,p in enumerate(adjacent_comb[j]):
#    plt.annotate(f'{i}',xy=p,fontsize=12)