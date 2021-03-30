import ipywidgets as widgets
import numpy as np
from bqplot import Figure, LinearScale, Axis, OrdinalScale, Scatter, Lines
from bqplot_image_gl.interacts import MouseInteraction
from matplotlib.colors import rgb2hex
from traitlets import HasTraits, observe, Any, link, Dict, Int, validate

from temnet.visualize.utils import get_colors_from_cmap


class TimeLine(HasTraits):
    data = Dict()
    num_frames = Int(0)
    current_index = Int(0)

    def __init__(self, x_scale=None, y_scale=None, axis_x=None, axis_y=None, height='70px', width='600px',
                 fig_margin=None, **kwargs):
        self._x_scale = x_scale or LinearScale(allow_padding=False)
        self._y_scale = y_scale or LinearScale(allow_padding=False)

        scales = {'x': self._x_scale, 'y': self._y_scale}

        self._axis_x = axis_x or Axis(scale=scales['x'])
        self._axis_y = axis_y or Axis(scale=scales['y'], orientation='vertical')

        fig_margin = fig_margin or {'top': 0, 'bottom': 20, 'left': 0, 'right': 0}

        self._figure = Figure(scales=scales, axes=[self._axis_x, self._axis_y],
                              fig_margin=fig_margin)

        self._figure.layout.height = height
        self._figure.layout.width = width

        self._index_indicator_y_scale = LinearScale(allow_padding=False)
        self._index_indicator = Lines(x=[0, 0], y=[0, 1], scales={'x': self._x_scale,
                                                                  'y': self._index_indicator_y_scale}, colors=['lime'])

        interaction = MouseInteraction(x_scale=self._x_scale,
                                       y_scale=self._index_indicator_y_scale,

                                       )

        def on_mouse_msg(_, change, __):
            if change['event'] in ('dragmove', 'click'):
                self.current_index = int(np.round(change['domain']['x']))

        self._figure.interaction = interaction
        interaction.events = ['click']
        self._figure.interaction.on_msg(on_mouse_msg)

        # self._next_button = widgets.Button(description='Next')
        #
        # def next(change):
        #     self.current_index = self.current_index + 1
        #
        # self._next_button.on_click(next)
        #
        # self._previous_button = widgets.Button(description='Previous')
        #
        # def previous(change):
        #     self.current_index = self.current_index - 1
        #
        # self._previous_button.on_click(previous)

        super().__init__(**kwargs)

    @validate('current_index')
    def _validate_current_index(self, change):
        value = change['value']
        value = max(value, 0)
        value = min(value, self.num_frames - 1)
        return value

    @observe('current_index')
    def _observe_current_index(self, change):
        self._index_indicator.x = [self.current_index] * 2

    @observe('data')
    def _observe_data(self, change):
        new_marks = []

        if len(self.data) == 0:
            return

        self.num_frames = len(self.data[list(self.data.keys())[0]])
        for key, values in self.data.items():
            x = np.arange(len(values), dtype=np.float)
            colors = [rgb2hex(i) for i in get_colors_from_cmap(values, cmap='Reds', vmin=0, vmax=1)]
            # print(np.random(len(x)))
            new_marks.append(
                Scatter(x=x, y=[1] * len(x),
                        scales={'x': self._x_scale,
                                'y': self._y_scale,
                                'rotation': LinearScale(min=0, max=180),
                                'size': LinearScale(min=0, max=1),
                                'skew': LinearScale(min=0, max=1)},
                        colors=colors,
                        skew=np.full(len(x), .5),
                        size=np.full(len(x), 2),
                        marker='rectangle', rotation=np.full(len(x), 180)))

        self._x_scale.max = self.num_frames

        new_marks += [self._index_indicator]
        self._figure.marks = new_marks

    @property
    def widget(self):
        return widgets.VBox([self._figure])


class Canvas(HasTraits):
    artists = Dict()
    tools = Dict()
    current_tool = Any()

    def __init__(self, x_scale=None, y_scale=None, axis_x=None, axis_y=None, height='600px', width='600px',
                 min_aspect_ratio=1, max_aspect_ratio=1, fig_margin=None, **kwargs):
        self._x_scale = x_scale or LinearScale(allow_padding=False)
        self._y_scale = y_scale or LinearScale(allow_padding=False)

        scales = {'x': self._x_scale, 'y': self._y_scale}

        self._axis_x = axis_x or Axis(scale=scales['x'])
        self._axis_y = axis_y or Axis(scale=scales['y'], orientation='vertical')

        fig_margin = fig_margin or {'top': 0, 'bottom': 50, 'left': 50, 'right': 0}

        self._figure = Figure(scales=scales, axes=[self._axis_x, self._axis_y],
                              min_aspect_ratio=min_aspect_ratio, max_aspect_ratio=max_aspect_ratio,
                              fig_margin=fig_margin)

        self._figure.layout.height = height
        self._figure.layout.width = width

        self._show_artists_panel = widgets.VBox([], layout=widgets.Layout(width='150px'))
        self._tool_panel = widgets.RadioButtons(options=[], layout=widgets.Layout(width='150px'))
        self._reset_view_button = widgets.Button(description='Fit view')

        self._reset_view_button.on_click(self.fit_view)

        self._left_panel = widgets.VBox([self._show_artists_panel, self._tool_panel, self._reset_view_button])
        self._tool_widgets = widgets.HBox([])

        super().__init__(**kwargs)

    def fit_view(self, _):
        xmin = np.min([artist.x_limits[0] for artist in self.artists.values()])
        xmax = np.max([artist.x_limits[1] for artist in self.artists.values()])
        ymin = np.min([artist.y_limits[0] for artist in self.artists.values()])
        ymax = np.max([artist.y_limits[1] for artist in self.artists.values()])
        extent = max(xmax - xmin, ymax - ymin)

        with self._x_scale.hold_sync(), self._y_scale.hold_sync():
            self._x_scale.min = xmin
            self._x_scale.max = xmin + extent
            self._y_scale.min = ymin
            self._y_scale.max = ymin + extent

    @observe('artists')
    def _observe_artists(self, change):
        self._figure.marks = []
        show_artist_widgets = []
        for key, artist in self.artists.items():
            artist.add_to_canvas(self)

            checkbox = widgets.Checkbox(value=True, description=key, indent=False, layout=widgets.Layout(width='90%'))
            link((checkbox, 'value'), (artist, 'visible'))
            show_artist_widgets.append(checkbox)

        self._show_artists_panel.children = show_artist_widgets

    @observe('tools')
    def _observe_tools(self, change):
        self._tool_panel.options = tuple(self.tools.keys())

        def change_tool(change):
            self.tools[change['old']].deactivate(self)
            self.tools[change['new']].activate(self)

        self._tool_panel.observe(change_tool, 'value')

        self.tools[self._tool_panel.options[0]].activate(self)

        tool_widgets = []
        for description, tool in self.tools.items():
            try:
                tool_widgets.append(tool.tool_widget(description))
            except AttributeError:
                pass

        self._tool_widgets = widgets.VBox(tool_widgets)

    @property
    def widget(self):
        return widgets.HBox([widgets.VBox([self._left_panel, self._tool_widgets]), self._figure, ])
