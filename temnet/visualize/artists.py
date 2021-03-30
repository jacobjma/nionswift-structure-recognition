import numpy as np
from bqplot import ColorScale, Scatter, Lines, LinearScale

from bqplot_image_gl import ImageGL
from traitlets import HasTraits, observe, Any, Bool, List
from traitlets import validate, link
from traittypes import Array
from matplotlib.colors import rgb2hex
from temnet.visualize.utils import get_colors_from_cmap
from temnet.visualize.extensions import link
import contextlib


class Artist:

    def __init__(self):
        pass


class ImageArtist(HasTraits):
    image = Any()
    visible = Bool()

    def __init__(self, **kwargs):
        self._mark = ImageGL(image=np.zeros((0, 0)))
        super().__init__(**kwargs)

    def add_to_canvas(self, canvas):
        try:
            limits = {'min': float(self.image.min()), 'max': float(self.image.max())}
        except ValueError:
            limits = {'min': 0, 'max': 1}

        color_scale = ColorScale(colors=['black', 'white'], **limits)
        self._mark.scales = {'x': canvas._x_scale, 'y': canvas._y_scale, 'image': color_scale}
        canvas._figure.marks = [self._mark] + canvas._figure.marks

    @property
    def x_limits(self):
        return [0, self.image.shape[1]]

    @property
    def y_limits(self):
        return [0, self.image.shape[0]]

    @observe('image')
    def _observe_image(self, change):
        image = self.image
        with self._mark.hold_sync():
            self._mark.x = [0, image.shape[0]]
            self._mark.y = [0, image.shape[1]]
            self._mark.image = image

            try:
                self._mark.scales['image'].min = float(image.min())
                self._mark.scales['image'].max = float(image.max())
            except (KeyError, ValueError):
                pass

    @validate('image')
    def _validate_image(self, proposal):
        try:
            image = proposal['value'].detach().cpu().numpy()
        except AttributeError:
            image = proposal['value']
        return image

    @observe('visible')
    def _observe_visible(self, change):
        self._mark.visible = self.visible


class PointArtist(HasTraits):
    points = Array(np.array((0, 2)))

    labels = Array()
    visible = Bool()
    updating = False

    def __init__(self, **kwargs):
        self._mark = Scatter(x=np.zeros((0,)), y=np.zeros((0,)), colors=['red'])

        def observe_x_and_y(*args):
            if self.updating:
                return
            with self._busy_updating():
                min_length = min(len(self._mark.x), len(self._mark.y))
                x = self._mark.x[:min_length]
                y = self._mark.y[:min_length]
                self.points = np.array([x, y]).T

        self._mark.observe(observe_x_and_y, ('x', 'y'))

        link((self, 'visible'), (self._mark, 'visible'))
        super().__init__(**kwargs)

    @contextlib.contextmanager
    def _busy_updating(self):
        self.updating = True
        try:
            yield
        finally:
            self.updating = False

    @observe('points')
    def _observe_points(self, *args):
        if self.updating:
            return
        with self._busy_updating():
            with self._mark.hold_sync():
                if len(self.points) > 0:
                    self._mark.x = self.points[:, 0]
                    self._mark.y = self.points[:, 1]
                else:
                    self._mark.x = np.zeros((0,))
                    self._mark.y = np.zeros((0,))

    def add_to_canvas(self, canvas):
        self._mark.scales = {'x': canvas._x_scale, 'y': canvas._y_scale}
        canvas._figure.marks = [self._mark] + canvas._figure.marks

    @property
    def x_limits(self):
        if len(self.points) == 0:
            return [0, 0]

        return [min(self.points[:, 0]), max(self.points[:, 1])]

    @property
    def y_limits(self):
        if len(self.points) == 0:
            return [0, 0]

        return [min(self.points[:, 0]), max(self.points[:, 1])]

    @observe('labels')
    def _observe_labels(self, change):
        with self._mark.hold_sync():
            colors = get_colors_from_cmap(self.labels, cmap='tab10', vmin=0, vmax=8)
            colors = [rgb2hex(color) for color in colors]
            self._mark.colors = colors


class LinesArtist(HasTraits):
    lines = List()
    visible = Bool()

    def __init__(self, colors=None, **kwargs):
        if colors is None:
            colors = ['red']

        scales = {'x': LinearScale(), 'y': LinearScale()}
        self._mark = Lines(x=np.zeros((0,)), y=np.zeros((0,)), scales=scales, colors=colors)
        link((self, 'visible'), (self._mark, 'visible'))
        super().__init__(**kwargs)

    def add_to_canvas(self, canvas):
        self._mark.scales = {'x': canvas._x_scale, 'y': canvas._y_scale}
        canvas._figure.marks = [self._mark] + canvas._figure.marks

    @observe('lines')
    def _observe_lines(self, *args):
        self._mark.x = [[point[0] for point in line] for line in self.lines]
        self._mark.y = [[point[1] for point in line] for line in self.lines]

    @property
    def x_limits(self):
        if len(self._mark.x) == 0:
            return [0, 0]

        try:
            return [min([min(line) for line in self._mark.x]), max([max(line) for line in self._mark.x])]
        except TypeError:
            return [min(self._mark.x), max(self._mark.x)]

    @property
    def y_limits(self):
        if len(self._mark.y) == 0:
            return [0, 0]

        try:
            return [min([min(line) for line in self._mark.y]), max([max(line) for line in self._mark.y])]
        except TypeError:
            return [min(self._mark.y), max(self._mark.y)]


class GraphArtist(LinesArtist):
    graph = Any()

    @observe('graph')
    def _observe_graph(self, change):
        lines = self.graph.points[self.graph.edges]
        self.lines = lines
