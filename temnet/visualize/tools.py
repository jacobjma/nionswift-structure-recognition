from abc import ABCMeta, abstractmethod

import ipywidgets as widgets
import numpy as np
from bqplot.interacts import BrushSelector
from bqplot.interacts import PanZoom as _PanZoom
from bqplot_image_gl.interacts import MouseInteraction
from scipy.spatial import KDTree
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from traitlets import HasTraits, Int, Any, Float, Bool, link


class Tool(metaclass=ABCMeta):

    @abstractmethod
    def activate(self, canvas):
        pass

    @abstractmethod
    def deactivate(self, canvas):
        pass


class PanZoom(Tool):

    def __init__(self, **kwargs):
        self._panzoom = _PanZoom(**kwargs)

    def activate(self, canvas):
        self._panzoom.scales = {'x': [canvas._x_scale], 'y': [canvas._y_scale]}
        canvas._figure.interaction = self._panzoom

    def deactivate(self, canvas):
        canvas._figure.interaction = None


class BoxZoom(Tool):

    def __init__(self, **kwargs):
        self._brush_selector = BrushSelector(**kwargs)

    def activate(self, canvas):
        self._brush_selector.x_scale = canvas._x_scale
        self._brush_selector.y_scale = canvas._y_scale

        def box_zoom(change):
            selected_x = self._brush_selector.selected_x
            selected_y = self._brush_selector.selected_y

            if (selected_x is not None) and (selected_x is not None):
                self._brush_selector.x_scale.min = min(selected_x[0], selected_x[1])
                self._brush_selector.x_scale.max = max(selected_x[0], selected_x[1])
                self._brush_selector.y_scale.min = min(selected_y[0], selected_y[1])
                self._brush_selector.y_scale.max = max(selected_y[0], selected_y[1])

                self._brush_selector.selected_x = None
                self._brush_selector.selected_y = None

        self._brush_selector.observe(box_zoom, 'brushing')
        canvas._figure.interaction = self._brush_selector

    def deactivate(self, canvas):
        canvas._figure.interaction = None


class DragPoint(HasTraits):
    point_artist = Any()

    def activate(self, canvas):
        self.point_artist._mark.enable_move = True

    def deactivate(self, canvas):
        self.point_artist._mark.enable_move = False


class AddPoint(HasTraits):
    label = Int(0)
    point_artist = Any()

    def activate(self, canvas):
        interaction = MouseInteraction(x_scale=self.point_artist._mark.scales['x'],
                                       y_scale=self.point_artist._mark.scales['y'])

        def on_mouse_msg(_, change, __):
            if change['event'] in ('click'):
                with self.point_artist._mark.hold_sync():
                    x = np.append(self.point_artist._mark.x, change['domain']['x'])
                    y = np.append(self.point_artist._mark.y, change['domain']['y'])
                    labels = np.append(self.point_artist.labels, self.label)

                    self.point_artist._mark.x = x
                    self.point_artist._mark.y = y
                    self.point_artist.labels = labels

        interaction.on_msg(on_mouse_msg)
        canvas._figure.interaction = interaction

    def deactivate(self, canvas):
        canvas._figure.interaction = None


class DeletePoint(HasTraits):
    point_artist = Any()
    tolerance = Float(50)

    def activate(self, canvas):
        interaction = MouseInteraction(x_scale=self.point_artist._mark.scales['x'],
                                       y_scale=self.point_artist._mark.scales['y'])

        def on_mouse_msg(_, change, __):

            if change['event'] in ('click'):
                with self.point_artist._mark.hold_sync():
                    query_point = [change['domain']['x'], change['domain']['y']]
                    points = np.array([self.point_artist._mark.x, self.point_artist._mark.y]).T
                    labels = self.point_artist.labels

                    tree = KDTree(points)
                    distance, index = tree.query(query_point, 1)
                    if distance < self.tolerance:
                        points = np.delete(points, index, axis=0)
                        labels = np.delete(labels, index, axis=0)

                        self.point_artist.points = points
                        self.point_artist.labels = labels

        interaction.on_msg(on_mouse_msg)
        canvas._figure.interaction = interaction
        # self.point_artist._mark.x = x
        # self.point_artist._mark.y = y
        # self.point_artist.labels = labels

    def deactivate(self, canvas):
        canvas._figure.interaction = None


class PolygonTool(HasTraits):
    lines_artist = Any()
    brush_size = Float(50)
    brush_sides = Int(10)
    subtract = Bool(False)

    def activate(self, canvas):
        interaction = MouseInteraction(x_scale=self.lines_artist._mark.scales['x'],
                                       y_scale=self.lines_artist._mark.scales['y'])

        def on_mouse_msg(_, change, __):
            if change['event'] in ('click', 'dragmove'):
                with self.lines_artist._mark.hold_sync():
                    center = [change['domain']['x'], change['domain']['y']]

                    i = np.arange(self.brush_sides, dtype=np.int32)
                    polygon = np.zeros((self.brush_sides, 2))
                    polygon[:, 0] = center[0] + self.brush_size * np.sin(i * 2 * np.pi / self.brush_sides)
                    polygon[:, 1] = center[1] + self.brush_size * np.cos(-i * 2 * np.pi / self.brush_sides)

                    if self.subtract:
                        union = cascaded_union([Polygon(line) for line in self.lines_artist.lines])
                        union = union.difference(Polygon(polygon))

                    else:
                        polygons = [Polygon(line) for line in self.lines_artist.lines] + [Polygon(polygon)]
                        union = cascaded_union(polygons)

                    try:
                        multigon = list(union)
                    except TypeError:
                        multigon = [union]

                    try:
                        self.lines_artist.lines = [list(polygon.exterior.coords) for polygon in multigon]
                    except AttributeError:
                        pass

        interaction.on_msg(on_mouse_msg)
        canvas._figure.interaction = interaction

    def deactivate(self, canvas):
        canvas._figure.interaction = None

    def tool_widget(self, description):

        slider = widgets.FloatSlider(value=10, max=50.0,
                                     description=description + ' brush size:',
                                     readout=False,
                                     layout=widgets.Layout(width='150px')
                                     )

        link((slider, 'value'), (self, 'brush_size'))

        return slider
