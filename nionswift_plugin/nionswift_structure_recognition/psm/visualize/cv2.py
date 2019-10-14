import cv2
import numpy as np


def add_faces(points, faces, image, colors):
    points = np.round(points).astype(int)

    for face, color in zip(faces, colors):
        cv2.fillConvexPoly(image, points[face], color)

    return image


def add_edges(points, edges, image, color, thickness=1):
    points = np.round(points).astype(int)
    for edge in edges:
        cv2.line(image, tuple(points[edge[0]][::-1]), tuple(points[edge[1]][::-1]), color=color,
                 thickness=thickness)

    return image


def add_points(points, image, size, color):
    points = np.round(points).astype(np.int)

    for points in points:
        cv2.circle(image, (points[1], points[0]), size, color, -1)

    return image


def scale_points_to_canvas(points, canvas, shape):
    points = points - [canvas[0], canvas[2]]
    points = points / [canvas[1] - canvas[0], canvas[3] - canvas[2]] * shape
    return points
