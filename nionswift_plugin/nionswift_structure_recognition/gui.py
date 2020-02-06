import gettext
import threading
import typing

import numpy as np
from nion.data import DataAndMetadata
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import DataItem as DataItemModule
from nion.swift.model import Symbolic
from nion.swift.Facade import Library, DataItem, Display
from nion.ui import Widgets

from .model import presets, build_model_from_dict
from .visualization import create_visualization

_ = gettext.gettext


def line_edit_template(ui, label, default_text=None, placeholder_text=None):
    row = ui.create_row_widget()
    row.add(ui.create_label_widget(label))
    row.add_spacing(5)
    widget = ui.create_line_edit_widget()
    row.add(widget)
    # row.add_spacing(5)
    widget._behavior.placeholder_text = placeholder_text
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
    # row.add_spacing(5)
    row.add_stretch()
    return row, widget


def check_box_template(ui, label):
    row = ui.create_row_widget()
    widget = ui.create_check_box_widget(label)
    # row.add_spacing(5)
    row.add(widget)
    return row, widget


class Unbinder:
    def __init__(self):
        self.__unbinders = list()
        self.__listener_map = dict()

    def close(self) -> None:
        for listener in self.__listener_map.values():
            listener.close()
        self.__listener_map = None

    def add(self, items, unbinders: typing.Sequence[typing.Callable[[], None]]) -> None:
        for item in items:
            if item and item not in self.__listener_map:
                self.__listener_map[item] = item.about_to_be_removed_event.listen(self.__unbind)
        self.__unbinders.extend(unbinders)

    def __unbind(self) -> None:
        for unbinder in self.__unbinders:
            unbinder()


class Section(Widgets.CompositeWidgetBase):

    def __init__(self, ui, section_id, section_title):
        self.__section_content_column = ui.create_column_widget()
        super().__init__(Widgets.SectionWidget(ui, section_title, self.__section_content_column,
                                               "inspector/" + section_id + "/open"))
        self.ui = ui  # for use in subclasses
        self._unbinder = Unbinder()

    def close(self) -> None:
        self._unbinder.close()
        super().close()

    def add_widget_to_content(self, widget):
        """Subclasses should call this to add content in the section's top level column."""
        self.__section_content_column.add_spacing(4)
        self.__section_content_column.add(widget)

    def finish_widget_content(self):
        """Subclasses should all this after calls to add_widget_content."""
        pass

    @property
    def _section_content_for_test(self):
        return self.__section_content_column


class SequencesSection(Section):

    def __init__(self, ui):
        super().__init__(ui, 'sequences', 'Sequences')

        analyse_sequence_row, self.analyse_sequence_push_button = push_button_template(ui, 'Analyse sequence')

        self.analyse_sequence_push_button.on_clicked = self.process_sequence
        self.add_widget_to_content(mask_weights_row)


    def set_preset(self, name):
        pass

    def update_parameters(self, parameters):
        pass


class ScaleDetectionSection(Section):

    def __init__(self, ui):
        super().__init__(ui, 'scale-detection', 'Scale Detection')
        lattice_constant_row, self.lattice_constant_line_edit = line_edit_template(self.ui, 'Lattice constant [Å]')
        min_sampling_row, self.min_sampling_line_edit = line_edit_template(self.ui, 'Min. sampling [Å / pixel]')
        crystal_system_row, self.crystal_system_combo_box = combo_box_template(self.ui, 'Crystal system', ['Hexagonal'])

        self.add_widget_to_content(crystal_system_row)
        self.add_widget_to_content(lattice_constant_row)
        self.add_widget_to_content(min_sampling_row)

    def set_preset(self, name):
        self.crystal_system_combo_box.current_item = presets[name]['scale']['crystal_system']
        self.lattice_constant_line_edit.text = presets[name]['scale']['lattice_constant']
        self.min_sampling_line_edit.text = presets[name]['scale']['min_sampling']

    def update_parameters(self, parameters):
        parameters['scale']['crystal_system'] = self.crystal_system_combo_box.current_item.lower()
        parameters['scale']['lattice_constant'] = float(self.lattice_constant_line_edit.text)
        parameters['scale']['min_sampling'] = float(self.min_sampling_line_edit.text)


class DeepLearningSection(Section):

    def __init__(self, ui):
        super().__init__(ui, 'deep-learning', 'Deep Learning')
        mask_weights_row, self.mask_weights_line_edit = line_edit_template(self.ui, 'Mask weights')
        density_weights_row, self.density_weights_line_edit = line_edit_template(self.ui, 'Density weights')
        training_scale_row, self.training_sampling_line_edit = line_edit_template(self.ui, 'Training sampling [A]')
        margin_row, self.margin_line_edit = line_edit_template(self.ui, 'Margin [A]')
        nms_distance_row, self.nms_distance_line_edit = line_edit_template(self.ui, 'NMS distance [A]')
        nms_threshold_row, self.nms_threshold_line_edit = line_edit_template(self.ui, 'NMS threshold')

        self.add_widget_to_content(mask_weights_row)
        self.add_widget_to_content(density_weights_row)
        self.add_widget_to_content(training_scale_row)
        self.add_widget_to_content(margin_row)
        self.add_widget_to_content(nms_distance_row)
        self.add_widget_to_content(nms_threshold_row)

    def set_preset(self, name):
        self.mask_weights_line_edit.text = presets[name]['mask_model']
        self.density_weights_line_edit.text = presets[name]['density_model']
        self.training_sampling_line_edit.text = presets[name]['training_sampling']
        self.margin_line_edit.text = presets[name]['margin']
        self.nms_distance_line_edit.text = presets[name]['nms']['distance']
        self.nms_threshold_line_edit.text = presets[name]['nms']['threshold']

    def update_parameters(self, parameters):
        parameters['training_sampling'] = float(self.training_sampling_line_edit.text)
        parameters['margin'] = float(self.margin_line_edit.text)
        parameters['mask_model'] = self.mask_weights_line_edit.text
        parameters['density_model'] = self.density_weights_line_edit.text
        parameters['nms']['distance'] = float(self.nms_distance_line_edit.text)
        parameters['nms']['threshold'] = float(self.nms_threshold_line_edit.text)


class VisualizationSection(Section):

    def __init__(self, ui):
        super().__init__(ui, 'visualization', 'Visualization')

        background_row, self.background_combo_box = combo_box_template(self.ui, 'Background',
                                                                       ['Image', 'Density', 'Segmentation', 'Solid'])

        points_row, self.points_check_box = check_box_template(self.ui, 'Overlay points')

        self.points_settings_column = self.ui.create_column_widget()
        self.points_color_column = self.ui.create_column_widget()
        self.points_size_line_edit = None
        self.points_color_combo_box = None
        self.point_color_solid_line_edit = None

        self.points_check_box.on_checked_changed = self.points_check_box_changed

        self.add_widget_to_content(background_row)
        self.add_widget_to_content(points_row)
        self.add_widget_to_content(self.points_settings_column)
        self.add_widget_to_content(self.points_color_column)

        # graph_row, self.graph_check_box = check_box_template(self.ui, 'Overlay graph')
        # section.column.add(graph_row)
        #
        # graph_settings_column = self.ui.create_column_widget()
        # self.line_width_line_edit = None
        # section.column.add(graph_settings_column)
        # section.column.add_spacing(5)
        #
        # def graph_check_box_changed(checked):
        #     graph_settings_column.remove_all()
        #
        #     if checked:
        #         line_width_row, self.line_width_line_edit = line_edit_template(self.ui, 'Line width', default_text=2)
        #
        #         graph_settings_column.add_spacing(5)
        #         graph_settings_column.add(line_width_row)
        #
        # self.graph_check_box.on_checked_changed = graph_check_box_changed
        # self.graph_check_box.checked = True
        # graph_check_box_changed(self.graph_check_box.checked)
        #
        # faces_row, self.faces_check_box = check_box_template(self.ui, 'Overlay faces')
        # section.column.add(faces_row)
        #
        # faces_settings_column = self.ui.create_column_widget()
        # self.faces_color_combo_box = None
        # self.faces_cmap_combo_box = None
        # section.column.add(faces_settings_column)
        # section.column.add_spacing(5)
        #
        # def faces_check_box_changed(checked):
        #     faces_settings_column.remove_all()
        #
        #     if checked:
        #         faces_color_row, self.faces_color_combo_box = combo_box_template(self.ui, 'Face color',
        #                                                                          ['Size', 'RMSD', 'exx', 'eyy'])
        #
        #         faces_settings_column.add_spacing(5)
        #         faces_settings_column.add(faces_color_row)
        #
        #         faces_cmap_row, self.faces_cmap_combo_box = combo_box_template(self.ui, 'Color map',
        #                                                                        ['gray', 'viridis', 'plasma', 'Paired',
        #                                                                         'tab10'])
        #
        #         faces_settings_column.add_spacing(5)
        #         faces_settings_column.add(faces_cmap_row)
        #
        #         # faces_vmin_row, self.faces_vmin_line_edit = line_edit_template(self.ui, 'vmin', default_text=0)
        #         # faces_vmax_row, self.faces_vmax_line_edit = line_edit_template(self.ui, 'vmin', default_text=0)
        #
        # self.faces_check_box.on_checked_changed = faces_check_box_changed
        # self.faces_check_box.checked = False
        # faces_check_box_changed(self.faces_check_box.checked)

    def points_check_box_changed(self, checked):
        self.points_settings_column.remove_all()
        self.points_color_column.remove_all()

        def point_color_combo_box_changed(item):
            self.points_color_column.remove_all()

            if item.lower() == 'solid':
                points_color_solid_row, self.point_color_solid_line_edit = line_edit_template(self.ui, 'Color',
                                                                                              default_text='red')
                self.points_color_column.add(points_color_solid_row)

        if checked:
            points_size_row, self.points_size_line_edit = line_edit_template(self.ui, 'Point size', default_text=3)

            points_color_row, self.points_color_combo_box = combo_box_template(self.ui, 'Point color',
                                                                               ['Solid', 'Class'])

            self.points_color_combo_box.on_current_item_changed = point_color_combo_box_changed

            self.points_settings_column.add_spacing(5)
            self.points_settings_column.add(points_size_row)
            self.points_settings_column.add(points_color_row)

            point_color_combo_box_changed(self.points_color_combo_box.current_item)

        else:
            self.points_size_line_edit = None

    def set_preset(self, name):
        self.points_check_box.checked = presets[name]['visualization']['points']['active']
        self.points_check_box_changed(self.points_check_box.checked)
        if self.points_check_box.checked:
            self.points_size_line_edit.text = presets[name]['visualization']['points']['size']

    def update_parameters(self, parameters):
        parameters['visualization']['background'] = self.background_combo_box.current_item.lower()
        parameters['visualization']['points']['active'] = self.points_check_box.checked

        if self.points_check_box.checked:
            parameters['visualization']['points']['size'] = int(self.points_size_line_edit.text)
            parameters['visualization']['points']['color_mode'] = self.points_color_combo_box.current_item.lower()

            if parameters['visualization']['points']['color_mode'] == 'solid':
                parameters['visualization']['points']['color'] = self.point_color_solid_line_edit.text

        # self.overlay_graph = self.graph_check_box.checked
        #
        # if self.overlay_graph:
        #     self.line_width = int(self.line_width_line_edit.text)
        #
        # self.overlay_faces = self.faces_check_box.checked
        #
        # if self.overlay_faces:
        #     self.faces_color = self.faces_color_combo_box.current_item.lower()
        #     self.faces_cmap = self.faces_cmap_combo_box.current_item


class StructureRecognitionPanel(Panel.Panel):

    def __init__(self, document_controller, panel_id, properties):
        super().__init__(document_controller, panel_id, _("Structure Recognition"))

        self.output_data_item = None
        self.source_data_item = None
        self.stop_live_analysis_event = threading.Event()
        self.stop_live_analysis_event.set()
        self.model = None
        self.parameters = presets['graphene']

        ui = document_controller.ui
        self.document_model = self.document_controller.document_model
        self.library = Library(self.document_model)

        main_column = ui.create_column_widget()
        scroll_area = ui.create_scroll_area_widget()
        scroll_area.content = main_column

        self.sequences_section = SequencesSection(ui)
        self.scale_detection_section = ScaleDetectionSection(ui)
        self.deep_learning_section = DeepLearningSection(ui)
        self.visualization_section = VisualizationSection(ui)

        def preset_combo_box_changed(x):
            self.scale_detection_section.set_preset(x.lower())
            self.deep_learning_section.set_preset(x.lower())
            # self.graph_module.set_preset(x.lower())
            self.visualization_section.set_preset(x.lower())

        preset_row, self.preset_combo_box = combo_box_template(ui, 'Preset', ['None', 'Graphene'], indent=True)
        self.preset_combo_box.on_current_item_changed = preset_combo_box_changed
        main_column.add(preset_row)

        run_row, self.run_push_button = push_button_template(ui, 'Start live analysis')

        def start_live_analysis():
            if self.stop_live_analysis_event.is_set():
                self.start_live_analysis()
                self.run_push_button.text = 'Abort live analysis'
            else:
                self.stop_live_analysis()
                self.run_push_button.text = 'Start live analysis'

        self.run_push_button.on_clicked = start_live_analysis

        main_column.add(run_row)
        main_column.add(self.scale_detection_section)
        main_column.add(self.deep_learning_section)
        main_column.add(self.visualization_section)
        self.widget = scroll_area
        main_column.add_stretch()

        preset_combo_box_changed('graphene')
        self.update_parameters()

        # def test(x):
        #    print(x)
        # document_controller.selected_display_panel
        # DisplayPanelManager


        # self.__cursor_changed_event_listener = self.document_controller.cursor_changed_event.listen(test)

    def create_display_computation(self, data_item):
        computation = self.document_model.create_computation()

        display_data_channel = self.document_model.get_display_item_for_data_item(data_item).display_data_channel
        specifier_dict = {"version": 1, "type": "data_source", "uuid": str(display_data_channel.uuid)}
        computation.create_object('input_data_item', specifier_dict)

        output_data_item = self.library.create_data_item_from_data(np.zeros_like(data_item.xdata.data[-2:]))

        computation.create_result('output_data_item', output_data_item.specifier.rpc_dict)
        computation.processing_id = 'structure-recognition.visualize'

        self.document_controller.document_model.append_computation(computation)

    def process_sequence(self):
        data_item = self.document_controller.selected_data_item

        images = data_item.xdata.data
        points = self.model.predict_batches(images)

        metadata = data_item.metadata
        metadata['struture-recognition'] = {}
        metadata['struture-recognition']['frames'] = {'{}'.format(i): p.tolist() for i, p in enumerate(points)}

        # visualization = create_visualization(images, None, None, points, self.parameters['visualization'])

        # dummy_data = np.zeros((1, 1, 3), dtype=np.uint8)
        # xdata = self.api.create_data_and_metadata(dummy_data, data_descriptor=descriptor)
        # self.output_data_item = self.api.library.create_data_item_from_data_and_metadata(xdata)

        # if source_data_item.xdata.is_sequence:
        #    current_index = source_data_item._DataItem__display_item.display_data_channel.sequence_index
        #    data = data[current_index]

    def get_camera(self):
        # return self.api.get_instrument_by_id("autostem_controller",version="1")

        camera = self.api.get_hardware_source_by_id('superscan', version='1')
        if camera is None:
            camera = self.api.get_hardware_source_by_id('usim_scan_device', '1.0')

        return camera
        # return self.api.get_hardware_source_by_id("nion1010",version="1")
        # return self.api.get_hardware_source_by_id('usim_scan_device', '1.0')

    def update_parameters(self):
        self.scale_detection_section.update_parameters(self.parameters)
        self.deep_learning_section.update_parameters(self.parameters)
        # self.graph_module.fetch_parameters()
        self.visualization_section.update_parameters(self.parameters)

        self.model = build_model_from_dict(self.parameters)

    def check_can_analyse_live(self):
        camera = self.get_camera()
        return camera.is_playing

    def start_live_analysis(self):
        # self.check_can_analyse_live()

        self.stop_live_analysis_event = threading.Event()
        self.process_live()

    def stop_live_analysis(self):
        self.stop_live_analysis_event.set()
        # self.thread.join()

    def process_live(self):
        pass


# if self.output_data_item is None:
#     descriptor = self.api.create_data_descriptor(is_sequence=False, collection_dimension_count=0, datum_dimension_count=2)
#
#     dummy_data = np.zeros((1, 1, 3), dtype=np.uint8)
#     xdata = self.api.create_data_and_metadata(dummy_data, data_descriptor=descriptor)
#     self.output_data_item = self.api.library.create_data_item_from_data_and_metadata(xdata)
#
# self.update_parameters()
#
# model = self.model
#
# print('start processing')
# with self.api.library.data_ref_for_data_item(self.output_data_item) as data_ref:
#
#     def thread_this(stop_live_analysis_event, camera, data_ref):
#         while not stop_live_analysis_event.is_set():
#             # if not camera.is_playing:
#             #    self.stop_live_analysis()
#
#             source_data = camera.grab_next_to_finish()  # TODO: This starts scanning? Must be a bug.
#
#             orig_images = source_data[0].data.copy()
#             points = model.predict(orig_images)
#
#             # self.output_data_item.title = 'Analysis of ' + camera.get_property_as_str('name')
#
#             visualization = create_visualization(orig_images, model.last_density, model.last_segmentation,
#                                                  points, self.parameters['visualization'])
#             # visualization = self.visualization_module.add_points(visualization, points)
#
#             data_ref.data = visualization
#
#     self.thread = threading.Thread(target=thread_this,
#                                    args=(self.stop_live_analysis_event, self.get_camera(), data_ref))
#     self.thread.start()


class VisualizeStructureRecognition:

    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, input_data_item):
        print('sssssss')
        sequence_index = input_data_item.data_item._DataItem__display_item.display_data_channel.sequence_index
        self.data = input_data_item.data_item.display_xdata.data

        # descriptor = DataAndMetadata.DataDescriptor(is_sequence=False,
        #                                             collection_dimension_count=0,
        #                                             datum_dimension_count=2)

        # self.xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=descriptor)

    def commit(self):
        self.computation.set_referenced_data('output_data_item', self.data)


workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(StructureRecognitionPanel, "structure-recognition-panel", _("Structure Recognition"),
                                 ["left", "right"], "right")

Symbolic.register_computation_type("structure-recognition.visualize", VisualizeStructureRecognition)
