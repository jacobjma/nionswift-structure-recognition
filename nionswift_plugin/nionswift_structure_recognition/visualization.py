import cv2
import matplotlib
import matplotlib.colors as mcolors
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


def add_faces(points, faces, image, colors):
    points = np.round(points).astype(int)
    for face, color in zip(faces, colors):
        cv2.fillConvexPoly(image, points[face][:, ::-1], tuple(map(int, color)))

    return image


def add_edges(image, points, edges, color, **kwargs):
    points = np.round(points).astype(int)
    for edge in edges:
        cv2.line(image, tuple(points[edge[0]]), tuple(points[edge[1]]), color=color, **kwargs)

    return image


def add_rectangles(image, rectangles, color, **kwargs):
    for rectangle in rectangles:
        cv2.rectangle(image,
                      (int(rectangle[0, 0]), int(rectangle[1, 0])),
                      (int(rectangle[0, 1]), int(rectangle[1, 1])), color=color, **kwargs)
    return image


def add_polygons(image, polygons, colors, thickness=3):
    default_colors = get_default_colors()
    array_colors = np.array(colors)

    if isinstance(colors, int):
        colors = [default_colors[colors]] * len(polygons)

    elif (len(array_colors.shape) == 1) & (array_colors.shape[0] == len(polygons)):
        colors = [default_colors[color] for color in colors]

    for polygon, color in zip(polygons, colors):
        polygon = np.array(polygon).astype(np.int32)[:, None]
        cv2.polylines(image, [polygon], True, color, thickness)
    return image


def add_text(image, text, upper_left_corner, color):
    (label_width, label_height), baseline = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                                                            thickness=1)
    org = (int(upper_left_corner[0]), int(upper_left_corner[1]) + label_height)
    _x1 = org[0]
    _y1 = org[1]
    _x2 = _x1 + label_width
    _y2 = org[1] - label_height
    image = cv2.rectangle(image, (_x1, _y1), (_x2, _y2), (255, 255, 255), cv2.FILLED)
    image = cv2.putText(image, text, org=org, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=color, thickness=1,
                        lineType=0)
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
        cv2.circle(image, (point[0], point[1]), size, color, -1)

    return image


def array_to_uint8_image(array):
    array = ((array - array.min()) / array.ptp() * 255).astype(np.uint8)
    return np.tile(array[..., None], (1, 1, 3))


def segmentation_to_uint8_image(segmentation):
    colors = np.array(get_default_colors(), dtype=np.uint8)
    painted_segmentation = colors[segmentation]
    return painted_segmentation
