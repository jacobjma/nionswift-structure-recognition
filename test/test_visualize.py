from nionswift_plugin.nionswift_structure_recognition.visualization import add_points, segmentation_to_uint8_image, \
    add_edges, add_rectangles
from psm.geometry import bounding_box_from_points
import matplotlib.pyplot as plt
import numpy as np
from psm.graph import stable_delaunay_graph


def test_add_points():
    points = np.random.rand(10, 2) * 100
    colors = np.random.randint(4, size=len(points))
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    add_points(image, points, colors)
    plt.imshow(image)
    plt.show()

    add_points(image, points, 3)
    plt.imshow(image)
    plt.show()


def test_visualize_segmentation():
    segmentation = np.random.randint(4, size=(100, 100))
    image = segmentation_to_uint8_image(segmentation)
    # plt.imshow(image)
    # plt.show()


def test_visualize_edges():
    points = np.random.rand(10, 2) * 100
    graph = stable_delaunay_graph(points, 2)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    add_edges(image, graph.points, graph.edges, (255, 0, 0))

    plt.imshow(image)
    plt.show()


def test_visualize_rectangles():
    points = np.random.rand(10, 2) * 100
    #graph = stable_delaunay_graph(points, 2)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    bbox = [bounding_box_from_points(points)]
    colors = np.random.randint(4, size=len(points))
    add_points(image, points, colors)
    add_rectangles(image, bbox, (255, 0, 0))

    plt.imshow(image)
    plt.show()

    # segmentation = np.random.randint(4, size=(100, 100))

    # image = segmentation_to_uint8_image(segmentation)
