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

    try:
        len(colors[0])
        individual_color = True
    except:
        individual_color = False

    if individual_color & (len(colors) != len(points)):
        raise RuntimeError()

    if individual_color:
        for points, color in zip(points, colors):
            cv2.circle(image, (points[0], points[1]), size, tuple(map(int, color)), -1)
    else:
        for point in points:
            cv2.circle(image, (point[0], point[1]), size, tuple(map(int, colors)), -1)

    return image


def float_images_to_rgb(images):
    images = ((images - np.min(images, axis=(-2, -1), keepdims=True)) /
             np.ptp(images, axis=(-2, -1), keepdims=True) * 255).astype(np.uint8)
    images = np.tile(images[..., None], (len(images.shape) * (1,)) + (3,))
    return images


def create_visualization(image, density, segmentation, points, parameters):
    if parameters['background'] == 'image':
        image = image

    elif parameters['background'] == 'density':
        image = density

    elif parameters['background'] == 'segmentation':
        image = segmentation

    elif parameters['background'] == 'solid':
        image = None

    else:
        raise RuntimeError()

    visualization = ((image - np.min(image, axis=(-2, -1), keepdims=True)) /
                     np.ptp(image, axis=(-2, -1), keepdims=True) * 255).astype(np.uint8)
    visualization = np.tile(visualization[..., None], (len(image.shape) * (1,)) + (3,))

    if parameters['points']['active']:
        if parameters['points']['color_mode'] == 'solid':
            color = mcolors.to_rgba(named_colors[parameters['points']['color']])[:3]
            colors = [tuple([int(x * 255) for x in color[::-1]])] * len(points)

        # elif self.points_color == 'class':
        #     colors = (get_colors_from_cmap(probabilities[:, 2], 'autumn', vmin=0, vmax=1) * 255).astype(int)
        #     colors = colors[:, :-1][:, ::-1]

        else:
            raise NotImplementedError()

        for i in range(len(points)):
            visualization[i] = add_points(points[i], visualization[i], parameters['points']['size'], colors[i])

    return visualization

    #
    # def add_edges(self, visualization):
    #     pass
    #     #return add_edges(graph.points, graph.edges(), visualization, (0, 0, 0), self.line_width)
    #
    # def add_faces(self, visualization):
    #     pass
    #     # if self.overlay_faces:
    #     #     if self.faces_color == 'size':
    #     #         colors = graph.faces().sizes()
    #     #         vmin = 0
    #     #         vmax = 10
    #     #
    #     #     elif self.faces_color == 'rmsd':
    #     #         colors = rmsd
    #     #         vmin = 0
    #     #         vmax = np.max(rmsd[rmsd != np.inf])
    #     #
    #     #     else:
    #     #         raise RuntimeError()
    #     #
    #     #     colors = (get_colors_from_cmap(colors, self.faces_cmap, vmin, vmax) * 255).astype(int)
    #     #
    #     #     visualization = add_faces(graph.points, graph.faces()[:-1], visualization, colors)
