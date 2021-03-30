import codecs
import contextlib
import json
import os

import ipywidgets as widgets
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon
from skimage.io import imread
from traitlets import HasTraits, Int, default, observe, Instance, Union, List, Unicode, Any, Float, link, Dict
from traitlets import TraitError
from traitlets.traitlets import _validate_link
from traittypes import Array

from temnet.psm.graph import stable_delaunay_graph
from temnet.utils import insert_folder_in_path


class link(object):
    updating = False

    def __init__(self, source, target, transform=None, hold_sync=None):
        _validate_link(source, target)
        self.source, self.target = source, target
        self._transform, self._transform_inv = (
            transform if transform else (lambda x: x,) * 2)

        self.hold_sync = hold_sync
        self.link()

    def link(self):
        try:
            setattr(self.target[0], self.target[1],
                    self._transform(getattr(self.source[0], self.source[1])))

        finally:
            self.source[0].observe(self._update_target, names=self.source[1])
            self.target[0].observe(self._update_source, names=self.target[1])

    @contextlib.contextmanager
    def _busy_updating(self):
        self.updating = True
        try:
            yield
        finally:
            self.updating = False

    def _update_target(self, change):
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.target[0], self.target[1], self._transform(change.new))
            if np.any(getattr(self.source[0], self.source[1]) != change.new):
                raise TraitError(
                    "Broken link {}: the source value changed while updating "
                    "the target.".format(self))

    def _update_source(self, change):
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.source[0], self.source[1],
                    self._transform_inv(change.new))
            if np.any(getattr(self.target[0], self.target[1]) != change.new):
                raise TraitError(
                    "Broken link {}: the target value changed while updating "
                    "the source.".format(self))

    def unlink(self):
        self.source[0].unobserve(self._update_target, names=self.source[1])
        self.target[0].unobserve(self._update_source, names=self.target[1])


class GaussianFilterWidget(HasTraits):
    image_in = Any()
    image_out = Any()
    sigma = Float(0.)

    @observe('image_in')
    def _observe_image_in(self, change):
        if self.sigma > 0.:
            self.image_out = gaussian_filter(self.image_in, self.sigma)
        else:
            self.image_out = self.image_in

    @observe('sigma')
    def _observe_sigma(self, change):
        self._observe_image_in(None)

    @property
    def widget(self):
        slider = widgets.FloatSlider(min=0, max=20, value=0, description='Gaussian filter')
        link((slider, 'value'), (self, 'sigma'))
        return slider


class ClampWidget(HasTraits):
    image_in = Any()
    image_out = Any()
    value = List()

    @default('value')
    def _default_value(self):
        return [0., 1.]

    @observe('image_in')
    def _observe_image_in(self, change):
        min_value = self.image_in.min()
        ptp = (self.image_in.max() - min_value)
        scaled = (self.image_in - min_value) / ptp
        image_out = torch.clip(scaled, min=self.value[0], max=self.value[1])
        self.image_out = image_out * ptp + min_value

    @observe('value')
    def _observe_sigma(self, change):
        self._observe_image_in(None)

    @property
    def widget(self):
        slider = widgets.FloatRangeSlider(min=0, max=1, value=[0., 1.], description='Clamp')
        link((slider, 'value'), (self, 'value'))
        return slider


class GraphWidget(HasTraits):
    points = Array(np.array((0, 2)))
    alpha = Float(2.)
    edges = List()

    @observe('points', 'alpha')
    def _observe_points(self, change):
        if (len(self.points) < 3):
            self.edges = []
            return

        graph = stable_delaunay_graph(self.points, alpha=self.alpha)
        self.edges = self.points[graph.edges].tolist()


class ImageFileLoader(HasTraits):
    path = Unicode()
    images = Any()
    num_frames = Int()

    @observe('path')
    def _observe_filename(self, change):
        _, ending = os.path.splitext(self.path)

        if ending.lower() in ('.tif', '.png', '.jpg'):
            images = imread(self.path)
        else:
            raise RuntimeError()

        if images.shape[-1] in (3, 4):
            images = np.swapaxes(images, 0, 2)

        if len(images.shape) == 2:
            images = images[None]

        assert len(images.shape) == 3

        self.images = images
        self.num_frames = len(self.images)


class TEMNetMetadataLoader(HasTraits):
    base_path = Unicode()
    path = Unicode()

    frame_data = List(allow_none=True)
    current_frame = Int(0)
    current_points = List(allow_none=True)
    current_labels = List(allow_none=True)
    current_segments = List(allow_none=True)

    segment_areas = Dict()
    summary = Unicode()

    # @validate('path')
    # def _validate_path(self, change):
    #     return

    @observe('path')
    def _observe_path(self, change):
        path = insert_folder_in_path(self.base_path, 'analysis', self.path)
        path = os.path.splitext(path)[0] + '.json'
        try:
            self.data = json.load(codecs.open(path, 'r', encoding='utf-8'))
            self.frame_data = self.data['frame_data']
        except FileNotFoundError:
            self.data = None
            self.frame_data = None
            self._observe_frame_data(None)

    @observe('frame_data', 'sampling')
    def _observe_frame_data(self, change):
        self._observe_current_frame(None)

        if self.data is None:
            summary = ''
            summary += f'fname, {self.path} \n'
            summary += f'analysed, False \n'
            summary += f'defect_in_sequence, {None} \n'
            summary += f'sampling, {None} \n'
            summary += f'num_frames, {None} \n'
            summary += f'first_defect_frame, {None} \n'
            summary += f'first_defect_x, {None} \n'
            summary += f'first_defect_y, {None} \n'
            summary += f'first_defect_type, {None} \n'

            self.summary = summary
            self.segment_areas = {'segment_areas': []}
            return

        sampling = np.median([f['sampling'] for f in self.frame_data])

        lattice_constant = 2.504
        side_length = lattice_constant
        area_per_vacancy = np.sqrt(3) / 4 * side_length ** 2 / sampling ** 2

        first_defect_frame = None
        segment_areas = []
        for i, frame in enumerate(self.frame_data):
            segments = frame['segments']
            segment_area = sum([Polygon(segment).area for segment in segments]) / area_per_vacancy

            if (segment_area > .5) & (first_defect_frame is None):
                first_defect_frame = i

            segment_areas.append(segment_area)

        self.segment_areas = {'segment_areas': segment_areas}

        summary = ''
        summary += f'fname, {self.path} \n'
        summary += f'analysed, True \n'
        summary += f'defect_in_sequence, {first_defect_frame is not None} \n'
        summary += f'sampling, {sampling:.5f} \n'
        summary += f'num_frames, {len(self.frame_data)} \n'
        summary += f'first_defect_frame, {first_defect_frame} \n'
        if first_defect_frame is not None:
            self.current_frame = first_defect_frame

            segments = self.frame_data[first_defect_frame]['segments']
            points = self.frame_data[first_defect_frame]['points']
            labels = self.frame_data[first_defect_frame]['labels']

            centers = np.array([tuple(Polygon(segment).centroid.coords)[0] for segment in segments])
            center = centers[np.argmin(centers[:, 1])]

            points = np.vstack(([center], points))
            graph = stable_delaunay_graph(points, 2)
            adjacent_labels = [labels[j - 1] for j in graph.adjacency[0]]

            if adjacent_labels == [1, 1, 1]:
                first_defect_type = 'B_vacancy'
            elif adjacent_labels == [0, 0, 0]:
                first_defect_type = 'N_vacancy'
            else:
                first_defect_type = 'unknown'

            summary += f'first_defect_x, {center[0]:.3f} \n'
            summary += f'first_defect_y, {center[1]:.3f} \n'
            summary += f'first_defect_type, {first_defect_type} \n'
        else:
            summary += f'first_defect_x, {None} \n'
            summary += f'first_defect_y, {None} \n'
            summary += f'first_defect_type, {None} \n'

        self.summary = summary

    @observe('current_frame')
    def _observe_current_frame(self, change):
        if self.frame_data is None:
            self.current_points = []
            self.current_points = []
            self.current_segments = []
        else:
            self.current_points = self.frame_data[self.current_frame]['points']
            self.current_labels = self.frame_data[self.current_frame]['labels']
            self.current_segments = self.frame_data[self.current_frame]['segments']

    def confirm_edits(self, *args):
        frame = self.frame_data[self.current_frame]
        frame['points'] = self.current_points
        frame['labels'] = self.current_labels
        frame['segments'] = self.current_segments
        self._observe_frame_data(None)

    @property
    def widget(self):
        confirm_button = widgets.Button(description='Confirm edits', layout=widgets.Layout(width='100px'))
        confirm_button.on_click(self.confirm_edits)
        input_file = widgets.Text(value='', description='Metadata file')
        save_button = widgets.Button(description='Write metadata', layout=widgets.Layout(width='100px'))
        link((self, 'path'), (input_file, 'value'))

        def save_data(*args):
            data = self.data
            data['frame_data'] = self.frame_data
            json.dump(data, codecs.open(self.path, 'w', encoding='utf-8'), separators=(',', ':'), indent=4)

        save_button.on_click(save_data)

        return widgets.VBox([widgets.HBox([input_file, confirm_button, save_button])])


class DictionaryPrint(HasTraits):
    data = Dict()
    description = Unicode()
    html = Any()

    @default('html')
    def _default_html(self):
        text = widgets.Textarea(
            value='',
            description=self.description,
            layout=widgets.Layout(width='400px', height='200px')
        )
        return text

    @observe('data')
    def _observe_data(self, change):
        self.html.value = str(self.data)

    @property
    def widget(self):
        button = widgets.Button(description='Write')
        return widgets.VBox([self.html, button])


class IntSliderWithButtons(widgets.HBox):
    value = Int(0)
    min = Int(0)
    max = Int(0)

    def __init__(self, **kwargs):
        self._slider = widgets.IntSlider(**kwargs)

        link((self._slider, 'value'), (self, 'value'))
        link((self._slider, 'min'), (self, 'min'))
        link((self._slider, 'max'), (self, 'max'))

        previous_button = widgets.Button(description='Previous')
        next_button = widgets.Button(description='Next')

        def next_item(*args):
            self._slider.value += 1

        def previous_item(*args):
            self._slider.value -= 1

        next_button.on_click(next_item)
        previous_button.on_click(previous_item)

        super().__init__([self._slider, previous_button, next_button])


class KeySelector(HasTraits):
    key = Any()
    dictionary = Dict()
    current_value = Any()

    @observe('current_index')
    def _observe_key(self):
        self.current_value = self.dictionary[self.key]


class ItemSelector(HasTraits):
    widget = Instance(IntSliderWithButtons)
    sequence = Union([Any(), List()])
    current_index = Int(0)
    current_item = Any()

    debounce = Float(0)
    _on_spillover_callback = Any()

    @default('current_item')
    def _default_selected(self):
        return self.sequence[0]

    @default('widget')
    def _default_widget(self):
        slider_with_buttons = IntSliderWithButtons(min=0, max=0, value=0)

        link((self, 'current_index'), (slider_with_buttons, 'value'))

        # def callback(*args):
        #     self.current_index = slider_with_buttons.value
        #
        # if self.debounce:
        #     slider_with_buttons.observe(throttle(self.debounce)(callback))
        # else:
        #     slider_with_buttons.observe(callback)

        return slider_with_buttons

    @observe('current_index')
    def _observe_current_index(self, change):
        self.current_item = self.sequence[self.current_index]

    @observe('sequence')
    def _observe_sequence(self, change):
        self.widget.max = len(self.sequence) - 1
        # link((self.widget, 'value'), (self, 'current_index'))

        clipped_current_index = min(self.current_index, self.widget.max)
        trigger = clipped_current_index == self.current_index

        self.current_index = clipped_current_index
        if trigger:
            self._observe_current_index(None)

    # @property
    # def widget(self):
    #     return self._slider
