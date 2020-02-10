import functools
import gettext
import threading
import typing

import cv2
import numpy as np
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.Facade import Library
from nion.swift.model import Symbolic
from nion.ui import Widgets
from nion.utils import Geometry
from scipy.spatial import KDTree

from .model import presets, build_model_from_dict
from .visualization import float_images_to_rgb

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


class Section(Widgets.CompositeWidgetBase):

    def __init__(self, main, section_id, section_title):
        self.main = main
        self.ui = self.main.ui
        self.__section_content_column = self.ui.create_column_widget()
        super().__init__(Widgets.SectionWidget(self.ui, section_title, self.__section_content_column,
                                               'structure-recognition/' + section_id + '/open'))

    def close(self) -> None:
        super().close()

    def add_widget_to_content(self, widget):
        """Subclasses should call this to add content in the section's top level column."""
        self.__section_content_column.add_spacing(4)
        self.__section_content_column.add(widget)


class SequencesSection(Section):

    def __init__(self, main):
        super().__init__(main, 'sequences', 'Sequences')

        indices_row, self.indices_line_edit = line_edit_template(self.ui, 'Sequence indices', placeholder_text='all')
        analyse_row, analyse_push_button = push_button_template(self.ui, 'Analyse sequence')
        analyse_push_button.on_clicked = main.process_sequence

        new_visualization_row, new_visualization_push_button = push_button_template(self.ui, 'New visualization')
        # export_row, self.export_push_button = push_button_template(ui, 'Export')

        previous_row, previous_push_button = push_button_template(self.ui, 'Previous')
        next_row, next_push_button = push_button_template(self.ui, 'Next')
        row = self.ui.create_row_widget()
        row.add(previous_row)
        row.add(next_row)

        new_visualization_push_button.on_clicked = main.new_visualization
        next_push_button.on_clicked = main.next_frame
        previous_push_button.on_clicked = main.previous_frame

        # self.analyse_sequence_push_button.on_clicked = self.process_sequence
        self.add_widget_to_content(indices_row)
        self.add_widget_to_content(analyse_row)
        self.add_widget_to_content(new_visualization_row)
        self.add_widget_to_content(row)
        # self.add_widget_to_content(export_row)

    def set_preset(self, name):
        pass

    def update_parameters(self, parameters):
        pass


class EditSection(Section):

    def __init__(self, main):
        super().__init__(main, 'edit', 'Edit')
        self.start_apply_cancel_edits_row = self.ui.create_row_widget()
        select_row, self.select_push_button = push_button_template(self.ui, 'Select')
        clear_selection_row, self.clear_selection_push_button = push_button_template(self.ui, 'Clear selection')
        delete_selected_row, self.delete_selected_push_button = push_button_template(self.ui, 'Delete selected')

        self.select_push_button.on_clicked = self.start_selecting
        self.clear_selection_push_button.on_clicked = self.clear_selection
        self.delete_selected_push_button.on_clicked = self.clear_selection

        # self.apply_or_cancel_edits_ui_change()
        self.add_widget_to_content(self.start_apply_cancel_edits_row)
        self.add_widget_to_content(self.select_push_button)
        self.add_widget_to_content(self.clear_selection_push_button)
        self.add_widget_to_content(self.delete_selected_push_button)

        self.apply_or_cancel_edits_ui_change()

    def apply_or_cancel_edits_ui_change(self):
        self.start_apply_cancel_edits_row.remove_all()
        start_editing_row, start_editing_push_button = push_button_template(self.ui, 'Start editing')
        self.start_apply_cancel_edits_row.add(start_editing_push_button)

        self.select_push_button.enabled = False
        self.clear_selection_push_button.enabled = False
        self.delete_selected_push_button.enabled = False

        start_editing_push_button.on_clicked = self.start_editing

    def start_edits_ui_change(self):
        self.start_apply_cancel_edits_row.remove_all()
        apply_edits_row, apply_edits_push_button = push_button_template(self.ui, 'Apply edits')
        cancel_edits_row, cancel_edits_push_button = push_button_template(self.ui, 'Cancel edits')

        self.start_apply_cancel_edits_row.add(apply_edits_push_button)
        self.start_apply_cancel_edits_row.add(cancel_edits_push_button)

        self.select_push_button.enabled = True
        self.clear_selection_push_button.enabled = True
        self.delete_selected_push_button.enabled = False

        apply_edits_push_button.on_clicked = self.apply_edits
        cancel_edits_push_button.on_clicked = self.cancel_edits

    def start_editing(self):
        self.edited_data_item = self.main.get_selected_data_item()
        self.main.edited_metadata = self.edited_data_item.metadata
        self.start_edits_ui_change()

    def cancel_edits(self):
        self.edited_data_item = None
        self.main.edited_metadata = None
        self.apply_or_cancel_edits_ui_change()

    def apply_edits(self):
        self.apply_or_cancel_edits_ui_change()

    def _overload_on_mouse_clicked(self, func):
        original_mouse_clicked = self.main.document_controller.selected_display_panel.canvas_widget.on_mouse_clicked

        def on_mouse_clicked(x, y, modifiers):
            original_mouse_clicked(x, y, modifiers)
            display_panel = self.main.document_controller.selected_display_panel
            canvas_item = display_panel.root_container._RootCanvasItem__mouse_tracking_canvas_item
            if canvas_item:
                canvas_item_point = display_panel.root_container.map_to_canvas_item(Geometry.IntPoint(y=y, x=x),
                                                                                    canvas_item)
                if hasattr(canvas_item, 'map_widget_to_image'):
                    image_point = canvas_item.map_widget_to_image(canvas_item_point)
                    func(image_point)

        self.main.document_controller.selected_display_panel.canvas_widget.on_mouse_clicked = on_mouse_clicked

    def clear_selection(self):
        if self.main.get_selected_data_item() is not self.edited_data_item:
            return

        for i in self.main.edited_metadata['structure-recognition']['frames'].keys():
            for j in self.main.edited_metadata['structure-recognition']['frames'][i].keys():
                self.main.edited_metadata['structure-recognition']['frames'][i][j]['selected'] = False

        self.main.update_visualization()

    def start_selecting(self):
        if self.main.get_selected_data_item() is not self.edited_data_item:
            return

        @functools.lru_cache(maxsize=1)
        def calculate_kd_tree_for_frame(sequence_index):
            points_dict = self.main.edited_metadata['structure-recognition']['frames'][str(sequence_index)]
            points_array = np.zeros((len(points_dict), 2))
            for i, point in enumerate(points_dict.values()):
                points_array[i] = point['position'][::-1]
            return KDTree(points_array)

        def func(x):
            current_data_item = self.main.get_selected_data_item()

            if (current_data_item is self.edited_data_item):
                sequence_index = self.main.get_sequence_index(current_data_item)
                tree = calculate_kd_tree_for_frame(sequence_index)
                _, idx = tree.query(x, 1)

                self.main.edited_metadata['structure-recognition']['frames'][str(sequence_index)][str(idx)][
                    'selected'] = True

                self.main.update_visualization()

        self._overload_on_mouse_clicked(func)

    def stop_editing(self):
        self.cursor_changed_event_listener = None  # TODO : Actually terminate the thread

    def set_preset(self, name):
        pass

    def update_parameters(self, parameters):
        pass


class ScaleDetectionSection(Section):

    def __init__(self, main):
        super().__init__(main, 'scale-detection', 'Scale Detection')
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

    def __init__(self, main):
        super().__init__(main, 'deep-learning', 'Deep Learning')
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
        self.mask_weights_line_edit.text = presets[name]['deep_learning']['mask_model']
        self.density_weights_line_edit.text = presets[name]['deep_learning']['density_model']
        self.training_sampling_line_edit.text = presets[name]['deep_learning']['training_sampling']
        self.margin_line_edit.text = presets[name]['deep_learning']['margin']
        self.nms_distance_line_edit.text = presets[name]['nms']['distance']
        self.nms_threshold_line_edit.text = presets[name]['nms']['threshold']

    def update_parameters(self, parameters):
        parameters['deep_learning']['training_sampling'] = float(self.training_sampling_line_edit.text)
        parameters['deep_learning']['margin'] = float(self.margin_line_edit.text)
        parameters['deep_learning']['mask_model'] = self.mask_weights_line_edit.text
        parameters['deep_learning']['density_model'] = self.density_weights_line_edit.text
        parameters['nms']['distance'] = float(self.nms_distance_line_edit.text)
        parameters['nms']['threshold'] = float(self.nms_threshold_line_edit.text)


class VisualizationSection(Section):

    def __init__(self, main):
        super().__init__(main, 'visualization', 'Visualization')

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

        self.mouse_thread = None
        self.live_thread = None
        self.process_thread = None
        self.mouse_position = None

        self.stop_mouse_event = threading.Event()

        self.stop_live_event = threading.Event()
        self.stop_live_event.set()

        self.visualization_data_items = {}
        self.visualized_data_items = {}

        self.model = None
        self.parameters = presets['graphene']

        ui = document_controller.ui
        self.document_model = self.document_controller.document_model
        self.library = Library(self.document_model)

        main_column = ui.create_column_widget()
        scroll_area = ui.create_scroll_area_widget()
        scroll_area.content = main_column

        preset_row, self.preset_combo_box = combo_box_template(ui, 'Preset', ['None', 'Graphene'], indent=True)
        run_row, self.run_push_button = push_button_template(ui, 'Start live analysis')

        def start_live_analysis():
            if self.stop_live_analysis_event.is_set():
                self.start_live_analysis()
                self.run_push_button.text = 'Abort live analysis'
            else:
                self.stop_live_analysis()
                self.run_push_button.text = 'Start live analysis'

        self.run_push_button.on_clicked = start_live_analysis

        self.scale_detection_section = ScaleDetectionSection(self)
        self.deep_learning_section = DeepLearningSection(self)
        self.visualization_section = VisualizationSection(self)
        self.sequences_section = SequencesSection(self)
        self.edit_section = EditSection(self)
        self.edited_metadata = None

        def preset_combo_box_changed(x):
            self.scale_detection_section.set_preset(x.lower())
            self.deep_learning_section.set_preset(x.lower())
            # self.graph_module.set_preset(x.lower())
            self.visualization_section.set_preset(x.lower())

        self.preset_combo_box.on_current_item_changed = preset_combo_box_changed

        main_column.add(preset_row)
        main_column.add(run_row)
        main_column.add(self.scale_detection_section)
        main_column.add(self.deep_learning_section)
        main_column.add(self.visualization_section)
        main_column.add(self.sequences_section)
        main_column.add(self.edit_section)
        self.widget = scroll_area
        main_column.add_stretch()

        preset_combo_box_changed('graphene')
        self.update_parameters()

        # import cv2

    def new_visualization(self):
        data_item = self.document_controller.selected_data_item
        new_visualization_data_item = self.library.create_data_item_from_data(
            np.zeros(data_item.data.shape[-2:] + (3,), dtype=np.uint8))
        self.visualization_data_items[data_item] = new_visualization_data_item
        self.visualized_data_items[new_visualization_data_item._data_item] = data_item
        self.update_visualization()

    def get_selected_data_item(self):
        try:
            return self.visualized_data_items[self.document_controller.selected_data_item]
        except KeyError:
            return self.document_controller.selected_data_item

    def get_visualizing_data_item(self, data_item):
        try:
            return self.visualization_data_items[data_item]
        except KeyError:
            return None

    def get_sequence_index(self, data_item):
        return self.document_model.get_display_item_for_data_item(data_item).display_data_channel.sequence_index

    def next_frame(self):
        data_item = self.get_selected_data_item()
        display_data_channel = self.document_model.get_display_item_for_data_item(data_item).display_data_channel
        display_data_channel.sequence_index += 1
        self.update_visualization()

    def previous_frame(self):
        data_item = self.get_selected_data_item()
        display_data_channel = self.document_model.get_display_item_for_data_item(data_item).display_data_channel
        display_data_channel.sequence_index -= 1
        self.update_visualization()

    def update_visualization(self):
        data_item = self.get_selected_data_item()
        visualizing_data_item = self.get_visualizing_data_item(data_item)

        if visualizing_data_item is None:
            return

        sequence_index = self.get_sequence_index(data_item)
        data = float_images_to_rgb(data_item.data[sequence_index])

        if self.edited_metadata is None:
            points = data_item.metadata['structure-recognition']['frames'][str(sequence_index)]
        else:
            points = self.edited_metadata['structure-recognition']['frames'][str(sequence_index)]

        for i, point in enumerate(points.values()):
            position = np.round(point['position']).astype(np.int)

            if point['selected']:
                cv2.circle(data, (position[0], position[1]), 3 + 1, (0, 255, 0), -1)

            cv2.circle(data, (position[0], position[1]), 3, (0, 0, 255), -1)

        with self.library.data_ref_for_data_item(visualizing_data_item) as data_ref:
            data_ref.data = data

    def export_metadata(self):
        pass

        # with Listener(on_click=on_click) as listener:
        #    listener.join()

        # print(self.document_controller.selected_display_panel)
        # self.x.value = 5

    def create_display_computation(self):
        data_item = self.document_controller.selected_data_item

        # computation = self.document_model.create_computation()

        # display_data_channel = self.document_model.get_display_item_for_data_item(data_item).display_data_channel
        # specifier_dict = {"version": 1, "type": "data_source", "uuid": str(display_data_channel.uuid)}
        # computation.create_object('input_data_item', specifier_dict)
        # # self.x = computation.create_variable(name='x', value_type='integral', value=70)
        #
        # output_data_item = self.library.create_data_item_from_data(
        #     np.zeros(data_item.xdata.data.shape[-2:] + (3,), dtype=np.uint8))
        #
        # computation.create_result('output_data_item', output_data_item.specifier.rpc_dict)
        # computation.processing_id = 'structure-recognition.visualize'
        #
        # self.document_controller.document_model.append_computation(computation)

    def process_sequence(self):
        data_item = self.document_controller.selected_data_item

        images = data_item.xdata.data
        points = self.model.predict_batches(images)

        metadata = data_item.metadata
        metadata['structure-recognition'] = {}
        metadata['structure-recognition']['frames'] = {}

        for i in range(len(points)):
            metadata['structure-recognition']['frames'][str(i)] = {}
            for j, point in enumerate(points[i].tolist()):
                metadata['structure-recognition']['frames'][str(i)][str(j)] = {}
                metadata['structure-recognition']['frames'][str(i)][str(j)]['position'] = point
                metadata['structure-recognition']['frames'][str(i)][str(j)]['label'] = 0
                metadata['structure-recognition']['frames'][str(i)][str(j)]['selected'] = False

        data_item.metadata = metadata

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
        print('execute')
        sequence_index = input_data_item.data_item._DataItem__display_item.display_data_channel.sequence_index
        image = input_data_item.data_item.display_xdata.data

        # points = input_data_item.data_item.metadata['structure-recognition']['frames'][str(sequence_index)]

        image = ((image - np.min(image, axis=(-2, -1), keepdims=True)) /
                 np.ptp(image, axis=(-2, -1), keepdims=True) * 255).astype(np.uint8)
        self.image = np.tile(image[..., None], (len(image.shape) * (1,)) + (3,))

        # for point in points.values():
        #    position = np.round(point['position']).astype(np.int)
        #    cv2.circle(self.image, (position[0], position[1]), 3, (0, 0, 255), -1)

        # self.image = add_points(points, image, 3, (0, 0, 255))

        # descriptor = DataAndMetadata.DataDescriptor(is_sequence=False,
        #                                             collection_dimension_count=0,
        #                                             datum_dimension_count=2)

        # self.xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=descriptor)

    def commit(self):
        self.computation.set_referenced_data('output_data_item', self.image)


workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(StructureRecognitionPanel, "structure-recognition-panel", _("Structure Recognition"),
                                 ["left", "right"], "right")

Symbolic.register_computation_type("structure-recognition.visualize", VisualizeStructureRecognition)
