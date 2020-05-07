import gettext
import threading

import numpy as np

# from .dl import DeepLearningModule
# # from .graph import GraphModule
# from .scale import ScaleDetectionModule
from .model import load_preset_model
from .visualization import mcolors, named_colors, add_points
# from .widgets import ScrollArea, push_button_template, combo_box_template

from .scale import find_hexagonal_sampling

_ = gettext.gettext

from nion.ui import Widgets


def line_edit_template(ui, label, default_text=None, placeholder_text=None):
    row = ui.create_row_widget()
    row.add(ui.create_label_widget(label))
    row.add_spacing(5)
    widget = ui.create_line_edit_widget()
    row.add(widget)
    row.add_spacing(5)
    # widget._widget._behavior.placeholder_text = placeholder_text
    widget.text = default_text
    return row, widget


def push_button_template(ui, label, callback=None):
    row = ui.create_row_widget()
    widget = ui.create_push_button_widget(label)
    widget.on_clicked = callback
    row.add(widget)
    return row, widget


def combo_box_template(ui, label, items, indent=False):
    row = ui.create_row_widget()
    if indent:
        row.add_spacing(8)
    row.add(ui.create_label_widget(label))
    row.add_spacing(5)
    widget = ui.create_combo_box_widget(items=items)
    row.add(widget)
    # row.add_stretch()
    return row, widget


def check_box_template(ui, label):
    row = ui.create_row_widget()
    widget = ui.create_check_box_widget(label)
    # row.add_spacing(5)
    row.add(widget)
    return row, widget


class ScrollArea:

    def __init__(self, ui):
        self.__ui = ui
        self.__scroll_area_widget = ui.create_scroll_area_widget()

    @property
    def _ui(self):
        return self.__ui

    @property
    def _widget(self):
        return self.__scroll_area_widget

    @property
    def content(self):
        return self._widget.content

    @content.setter
    def content(self, value):
        self._widget.content = value


class Section:

    def __init__(self, ui, title):
        self.__ui = ui
        self.__section_content_column = self.__ui._ui.create_column_widget()
        self.__section_widget = Widgets.SectionWidget(self.__ui._ui, title, self.__section_content_column, 'test')
        self.column = ui.create_column_widget()
        self.__section_content_column.add(self.column._widget)

    @property
    def _ui(self):
        return self.__ui

    @property
    def _widget(self):
        return self.__section_widget

    @property
    def _section_content_column(self):
        return self.__section_content_column


class StructureRecognitionModule(object):

    def __init__(self, ui, document_controller):
        self.ui = ui
        self.document_controller = document_controller

    def create_widgets(self, column):
        raise NotImplementedError()

    def set_preset(self, name):
        raise NotImplementedError()

    def fetch_parameters(self):
        raise NotImplementedError()


class ScaleDetectionModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

    def create_widgets(self, column):
        section = Section(self.ui, 'Scale detection')
        column.add(section)

        model_row, self.model_combo_box = combo_box_template(self.ui, 'Model',
                                                             ['Fourier-space Hexagonal', 'Real-space Hexagonal'])
        lattice_constant_row, self.lattice_constant_line_edit = line_edit_template(self.ui, 'Lattice constant [Å]')
        min_sampling_row, self.min_sampling_line_edit = line_edit_template(self.ui, 'Min. sampling [Å / pixel]')
        max_sampling_row, self.max_sampling_line_edit = line_edit_template(self.ui, 'Max. sampling [Å / pixel]')
        #calibrate_every_row, self.calibrate_every_check_box = check_box_template(self.ui, 'Calibrate every frame')
        #calibrate_every_row, self.calibrate_every_check_box = check_box_template(self.ui, 'Calibrate every frame')

        section.column.add(model_row)
        section.column.add(lattice_constant_row)
        section.column.add(min_sampling_row)
        section.column.add(max_sampling_row)
        #section.column.add(calibrate_every_row)

    def set_preset(self, name):
        self.model_combo_box.current_item = 'Fourier-space Hexagonal'  # presets[name]['scale']['model']
        self.lattice_constant_line_edit.text = 2.46  # presets[name]['scale']['lattice_constant']
        self.min_sampling_line_edit.text = .01  # presets[name]['scale']['min_sampling']
        self.max_sampling_line_edit.text = .1  # presets[name]['scale']['max_sampling']

    def fetch_parameters(self):
        self.model = self.model_combo_box._widget.current_item.lower()
        self.lattice_constant = float(self.lattice_constant_line_edit._widget.text)
        self.min_sampling = float(self.min_sampling_line_edit._widget.text)

    def detect_scale(self, image):
        if self.model not in ['fourier-space hexagonal',
                              'real-space hexagonal']:
            raise RuntimeError('model {} not recognized for scale recognition'.format(self.model))

        scale = find_hexagonal_sampling(image, lattice_constant=self.lattice_constant, min_sampling=self.min_sampling)
        return scale


class VisualizationModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

        self.training_sampling = None

    def create_widgets(self, column):
        section = Section(self.ui, 'Visualization')
        column.add(section)

        background_row, self.background_combo_box = combo_box_template(self.ui, 'Background', ['Image'])
        section.column.add(background_row)

        points_row, self.points_check_box = check_box_template(self.ui, 'Overlay points')
        section.column.add(points_row)

        points_settings_column = self.ui.create_column_widget()
        points_color_column = self.ui.create_column_widget()

        self.points_size_line_edit = None
        self.points_color_combo_box = None
        self.point_color_solid_line_edit = None

        section.column.add(points_settings_column)
        section.column.add(points_color_column)
        section.column.add_spacing(5)

        def points_check_box_changed(checked):

            points_settings_column._widget.remove_all()
            points_color_column._widget.remove_all()

            def point_color_combo_box_changed(item):
                points_color_column._widget.remove_all()

                if item.lower() == 'solid':
                    points_color_solid_row, self.point_color_solid_line_edit = line_edit_template(self.ui, 'Color',
                                                                                                  default_text='red')
                    points_color_column.add(points_color_solid_row)

            if checked:
                points_size_row, self.points_size_line_edit = line_edit_template(self.ui, 'Point size', default_text=3)

                points_color_row, self.points_color_combo_box = combo_box_template(self.ui, 'Point color',
                                                                                   ['Solid', 'Class'])

                self.points_color_combo_box.on_current_item_changed = point_color_combo_box_changed

                points_settings_column.add_spacing(5)
                points_settings_column.add(points_size_row)
                points_settings_column.add(points_color_row)

                point_color_combo_box_changed(self.points_color_combo_box.current_item)

            else:
                self.points_size_line_edit = None

        self.points_check_box.on_checked_changed = points_check_box_changed
        self.points_check_box.checked = True
        points_check_box_changed(self.points_check_box.checked)

    def set_preset(self, x):
        pass

    def fetch_parameters(self):
        self.background = self.background_combo_box._widget.current_item.lower()
        self.overlay_points = self.points_check_box.checked

        if self.overlay_points:
            self.points_size = int(self.points_size_line_edit.text)

            self.points_color = self.points_color_combo_box.current_item.lower()

            if self.points_color == 'solid':
                self.points_color_solid = self.point_color_solid_line_edit.text

    def create_background(self, image, density, segmentation):

        if self.background == 'image':
            visualization = ((image - image.min()) / image.ptp() * 255).astype(np.uint8)
            visualization = np.tile(visualization[..., None], (1, 1, 3))

        elif self.background == 'density':
            visualization = (density * 255).astype(np.uint8)
            visualization = np.tile(visualization[..., None], (1, 1, 3))

        elif self.background == 'segmentation':
            visualization = (segmentation / segmentation.max() * 255).astype(np.uint8)
            visualization = np.tile(visualization[..., None], (1, 1, 3))

        elif self.background == 'solid':
            visualization = None

        else:
            raise RuntimeError()

        return visualization

    def add_points(self, visualization, points):
        if self.overlay_points:
            if self.points_color == 'solid':
                color = mcolors.to_rgba(named_colors[self.points_color_solid])[:3]
                colors = [tuple([int(x * 255) for x in color[::-1]])] * len(points)

            else:
                raise NotImplementedError()

            visualization = add_points(points, visualization, self.points_size, colors)

        return visualization


class StructureRecognitionPanelDelegate:

    def __init__(self, api):
        self.api = api
        self.panel_id = "structure-recognition-panel"
        self.panel_name = _("Structure Recognition")
        self.panel_positions = ["left", "right"]
        self.panel_position = "right"
        self.output_data_item = None
        self.source_data_item = None
        self.stop_live_analysis_event = threading.Event()
        self.stop_live_analysis_event.set()
        self.model = None

    def create_panel_widget(self, ui, document_controller):
        self.ui = ui
        self.document_controller = document_controller
        main_column = ui.create_column_widget()
        scroll_area = ScrollArea(ui._ui)
        scroll_area.content = main_column._widget

        self.scale_detection_module = ScaleDetectionModule(ui, document_controller)
        # # self.graph_module = GraphModule(ui, document_controller)
        self.visualization_module = VisualizationModule(ui, document_controller)

        def preset_combo_box_changed(x):
            self.scale_detection_module.set_preset(x.lower())
            # # self.graph_module.set_preset(x.lower())
            self.visualization_module.set_preset(x.lower())

        preset_row, self.preset_combo_box = combo_box_template(self.ui, 'Preset', ['None', 'Graphene'],
                                                               indent=True)
        self.preset_combo_box.on_current_item_changed = preset_combo_box_changed
        main_column.add(preset_row)

        run_row, self.run_push_button = push_button_template(ui, 'Start live analysis')

        def start_live_analysis():
            if self.stop_live_analysis_event.is_set():
                self.start_live_analysis()
            else:
                self.stop_live_analysis()

        self.run_push_button.on_clicked = start_live_analysis

        # def stop_live_analysis():
        #    self.continue_live_analysis = False
        #
        # stop_row, stop_push_button = push_button_template(ui, 'Stop', stop_live_analysis)

        live_analysis_row = ui.create_row_widget()
        live_analysis_row.add(run_row)
        # live_analysis_row.add(stop_row)
        main_column.add(live_analysis_row)

        self.scale_detection_module.create_widgets(main_column)
        # self.deep_learning_module.create_widgets(main_column)
        # # self.graph_module.create_widgets(main_column)
        self.visualization_module.create_widgets(main_column)

        # self.preset_combo_box.current_item = 'Graphene'
        self.scale_detection_module.set_preset('graphene')

        main_column.add_stretch()

        return scroll_area

    def get_camera(self):
        camera = self.api.get_hardware_source_by_id("superscan", version="1")

        if camera is None:
            return self.api.get_hardware_source_by_id('usim_scan_device', '1.0')
        else:
            return camera

    def update_parameters(self):
        self.scale_detection_module.fetch_parameters()
        # # self.graph_module.fetch_parameters()
        self.visualization_module.fetch_parameters()

    def check_can_analyse_live(self):
        camera = self.get_camera()
        return camera.is_playing

    def start_live_analysis(self):
        # self.check_can_analyse_live()
        self.run_push_button.text = 'Abort live analysis'
        self.stop_live_analysis_event = threading.Event()
        self.process_live()

    def stop_live_analysis(self):
        self.run_push_button.text = 'Start live analysis'
        self.stop_live_analysis_event.set()
        # self.thread.join()

    def process_live(self):
        if self.model is None:
            self.model = load_preset_model('graphene')

        self.output_data_item = self.document_controller.library.create_data_item()
        self.output_data_item.title = 'Visualization'

        # if self.output_data_item is None:
        #     descriptor = self.api.create_data_descriptor(is_sequence=False, collection_dimension_count=0,
        #                                                  datum_dimension_count=2)
        #
        #     dummy_data = np.zeros((1, 1, 3), dtype=np.uint8)
        #     xdata = self.api.create_data_and_metadata(dummy_data, data_descriptor=descriptor)
        #     self.output_data_item = self.api.library.create_data_item_from_data_and_metadata(xdata)

        self.update_parameters()

        print('start processing')
        with self.api.library.data_ref_for_data_item(self.output_data_item) as data_ref:

            def thread_this(stop_live_analysis_event, camera, data_ref):
                while not stop_live_analysis_event.is_set():
                    if not camera.is_playing:
                        self.stop_live_analysis()

                    source_data = camera.grab_next_to_finish()  # TODO: This starts scanning? Must be a bug.

                    image = source_data[0].data.copy()

                    sampling = self.scale_detection_module.detect_scale(image)
                    output = self.model(image, sampling)

                    if output is not None:
                        visualization = self.visualization_module.create_background(image,
                                                                                    output['density'],
                                                                                    output['segmentation']
                                                                                    )

                        visualization = self.visualization_module.add_points(visualization, output['points'])

                        def update_data_item():
                            data_ref.data = visualization

                        self.api.queue_task(update_data_item)

            self.thread = threading.Thread(target=thread_this,
                                           args=(self.stop_live_analysis_event, self.get_camera(), data_ref))
            self.thread.start()


class StructureRecognitionExtension(object):
    extension_id = "nion.swift.extension.structure_recognition"

    def __init__(self, api_broker):
        api = api_broker.get_api(version='~1.0', ui_version='~1.0')
        self.__panel_ref = api.create_panel(StructureRecognitionPanelDelegate(api))

    def close(self):
        self.__panel_ref.close()
        self.__panel_ref = None
