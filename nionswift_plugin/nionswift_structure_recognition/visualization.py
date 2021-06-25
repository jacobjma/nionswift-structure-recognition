# import cv2
import matplotlib
import matplotlib.colors as mcolors
import numpy as np

from skimage import draw


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


def add_edges(image, points, edges, color, **kwargs):
    points = np.round(points).astype(int)
    for edge in edges:
        a, b = points[edge[0]], points[edge[1]]
        try:
            image[draw.line(a[1], a[0], b[1], b[0])] = color
        except IndexError:
            pass

    return image


def add_rectangles(image, rectangles, color, **kwargs):
    for rectangle in rectangles:
        image[draw.rectangle_perimeter((int(rectangle[0, 0]), int(rectangle[1, 0])),
                                       (int(rectangle[0, 1]), int(rectangle[1, 1])),
                                       shape=image.shape, clip=True)] = color

    return image


def add_polygons(image, polygons, color):
    for polygon in polygons:
        polygon = np.array(polygon).astype(np.int32)
        image[draw.polygon_perimeter(polygon[:, 1], polygon[:, 0], shape=image.shape, clip=True)] = color

    return image


def add_text(image, text, upper_left_corner, color):
    # (label_width, label_height), baseline = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
    #                                                         thickness=1)
    # org = (int(upper_left_corner[0]), int(upper_left_corner[1]) + label_height)
    # _x1 = org[0]
    # _y1 = org[1]
    # _x2 = _x1 + label_width
    # _y2 = org[1] - label_height
    # image = cv2.rectangle(image, (_x1, _y1), (_x2, _y2), (255, 255, 255), cv2.FILLED)
    # image = cv2.putText(image, text, org=org, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=color, thickness=1,
    #                     lineType=0)
    return image


def get_default_colors():
    colors = mcolors.BASE_COLORS
    colors = [colors[key][::-1] for key in ['r', 'b', 'y', 'g', 'c', 'm']]
    return [(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in colors]


def add_points(image, points, colors, size=4):
    points = np.round(points).astype(np.int)

    default_colors = get_default_colors()
    array_colors = np.array(colors)

    if isinstance(colors, int):
        colors = [default_colors[colors]] * len(points)

    elif (len(array_colors.shape) == 1) & (array_colors.shape[0] == len(points)):
        colors = [default_colors[color] for color in colors]

    for point, color in zip(points, colors):
        image[draw.disk((point[1], point[0]), size, shape=image.shape)] = color

    return image


def array_to_uint8_image(array):
    array = ((array - array.min()) / array.ptp() * 255).astype(np.uint8)
    return np.tile(array[..., None], (1, 1, 3))


def segmentation_to_uint8_image(segmentation):
    colors = np.array(get_default_colors(), dtype=np.uint8)
    return colors[segmentation]
