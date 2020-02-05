import gettext
import threading

import numpy as np

from .model import presets, build_model_from_dict
from .widgets import ScrollArea, push_button_template, combo_box_template, StructureRecognitionModule, \
    line_edit_template, Section, check_box_template
from .visualization import create_visualization

_ = gettext.gettext


class ScaleDetectionModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

        self.crystal_system = None
        self.lattice_constant = None

    def create_widgets(self, column):
        section = Section(self.ui, 'Scale detection')
        column.add(section)

        lattice_constant_row, self.lattice_constant_line_edit = line_edit_template(self.ui, 'Lattice constant [Å]')
        min_sampling_row, self.min_sampling_line_edit = line_edit_template(self.ui, 'Min. sampling [Å / pixel]')
        crystal_system_row, self.crystal_system_combo_box = combo_box_template(self.ui, 'Crystal system', ['Hexagonal'])

        section.column.add(crystal_system_row)
        section.column.add(lattice_constant_row)
        section.column.add(min_sampling_row)

    def set_preset(self, name):
        self.crystal_system_combo_box.current_item = presets[name]['scale']['crystal_system']
        self.lattice_constant_line_edit.text = presets[name]['scale']['lattice_constant']
        self.min_sampling_line_edit.text = presets[name]['scale']['min_sampling']

    def update_parameters(self, parameters):
        parameters['scale']['crystal_system'] = self.crystal_system_combo_box._widget.current_item.lower()
        parameters['scale']['lattice_constant'] = float(self.lattice_constant_line_edit._widget.text)
        parameters['scale']['min_sampling'] = float(self.min_sampling_line_edit._widget.text)


class DeepLearningModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

    def create_widgets(self, column):
        section = Section(self.ui, 'Deep learning')
        column.add(section)

        # model_row, self.model_line_edit = line_edit_template(self.ui, 'Model')
        mask_weights_row, self.mask_weights_line_edit = line_edit_template(self.ui, 'Mask weights')
        density_weights_row, self.density_weights_line_edit = line_edit_template(self.ui, 'Density weights')
        training_scale_row, self.training_sampling_line_edit = line_edit_template(self.ui, 'Training sampling [A]')
        margin_row, self.margin_line_edit = line_edit_template(self.ui, 'Margin [A]')
        nms_distance_row, self.nms_distance_line_edit = line_edit_template(self.ui, 'NMS distance [A]')
        nms_threshold_row, self.nms_threshold_line_edit = line_edit_template(self.ui, 'NMS threshold')

        # section.column.add(model_row)
        section.column.add(mask_weights_row)
        section.column.add(density_weights_row)
        section.column.add(training_scale_row)
        section.column.add(margin_row)
        section.column.add(nms_distance_row)
        section.column.add(nms_threshold_row)

    def set_preset(self, name):
        # self.model_line_edit.text = presets[name]['model_file']
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


class VisualizationModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

    def create_widgets(self, column):
        section = Section(self.ui, 'Visualization')
        column.add(section)

        background_row, self.background_combo_box = combo_box_template(self.ui, 'Background',
                                                                       ['Image', 'Density', 'Segmentation', 'Solid'])
        section.column.add(background_row)

        points_row, self.points_check_box = check_box_template(self.ui, 'Overlay points')
        section.column.add(points_row)

        self.points_settings_column = self.ui.create_column_widget()
        self.points_color_column = self.ui.create_column_widget()

        self.points_size_line_edit = None
        self.points_color_combo_box = None
        self.point_color_solid_line_edit = None

        section.column.add(self.points_settings_column)
        section.column.add(self.points_color_column)
        section.column.add_spacing(5)

        self.points_check_box.on_checked_changed = self.points_check_box_changed

        # graph_row, self.graph_check_box = check_box_template(self.ui, 'Overlay graph')
        # section.column.add(graph_row)
        #
        # graph_settings_column = self.ui.create_column_widget()
        # self.line_width_line_edit = None
        # section.column.add(graph_settings_column)
        # section.column.add_spacing(5)
        #
        # def graph_check_box_changed(checked):
        #     graph_settings_column._widget.remove_all()
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
        #     faces_settings_column._widget.remove_all()
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
        self.points_settings_column._widget.remove_all()
        self.points_color_column._widget.remove_all()

        def point_color_combo_box_changed(item):
            self.points_color_column._widget.remove_all()

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
        parameters['visualization']['background'] = self.background_combo_box._widget.current_item.lower()
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


from nion.swift import Panel


class StructureRecognitionPanel(Panel.Panel):

    # def __init__(self, api):
    def __init__(self, document_controller, panel_id, properties):
        super().__init__(document_controller, panel_id, _("Structure Recognition"))

        self.output_data_item = None
        self.source_data_item = None
        self.stop_live_analysis_event = threading.Event()
        self.stop_live_analysis_event.set()
        self.model = None
        self.parameters = presets['graphene']



        document_model = document_controller.document_model
        ui = document_controller.ui

        main_column = ui.create_column_widget()
        self.widget = main_column

        # self.api = api
        # self.panel_id = "structure-recognition-panel"
        # self.panel_name = _("Structure Recognition")
        # self.panel_positions = ["left", "right"]
        # self.panel_position = "right"

        # super().__init__(api._document_controller, panel_id, _("Structure"))

    #def create_panel_widget(self, ui, document_controller):
        #self.ui = ui
        # self.document_controller = document_controller

        main_column = ui.create_column_widget()
        scroll_area = ScrollArea(ui._ui)
        scroll_area.content = main_column._widget

        self.scale_detection_module = ScaleDetectionModule(ui, document_controller)
        self.deep_learning_module = DeepLearningModule(ui, document_controller)
        # self.graph_module = GraphModule(ui, document_controller)
        self.visualization_module = VisualizationModule(ui, document_controller)

        def preset_combo_box_changed(x):
            self.scale_detection_module.set_preset(x.lower())
            self.deep_learning_module.set_preset(x.lower())
            # self.graph_module.set_preset(x.lower())
            self.visualization_module.set_preset(x.lower())

        preset_row, self.preset_combo_box = combo_box_template(ui, 'Preset', ['None', 'Graphene'],
                                                               indent=True)
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

        analyse_sequence_row, self.analyse_sequence_push_button = push_button_template(ui, 'Analyse sequence')

        self.analyse_sequence_push_button.on_clicked = self.process_sequence

        main_column.add(run_row)
        main_column.add(analyse_sequence_row)

        self.scale_detection_module.create_widgets(main_column)
        self.deep_learning_module.create_widgets(main_column)
        # self.graph_module.create_widgets(main_column)
        self.visualization_module.create_widgets(main_column)

        self.preset_combo_box.current_item = 'Graphene'
        main_column.add_stretch()
        self.update_parameters()
        #return scroll_area

    def process_sequence(self):
        window = self.api.application.document_windows[0]
        data_item = window.target_data_item
        images = data_item.xdata.data[:2]

        points = self.model.predict(images)

        visualization = create_visualization(images, None, None, points, self.parameters['visualization'])

        descriptor = self.api.create_data_descriptor(is_sequence=data_item.xdata.is_sequence,
                                                     collection_dimension_count=0,
                                                     datum_dimension_count=2)

        # data = (np.random.rand(4, 64, 64, 3) * 255).astype(np.uint8)
        xdata = self.api.create_data_and_metadata(visualization, data_descriptor=descriptor)
        self.api.library.create_data_item_from_data_and_metadata(xdata)

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
        self.scale_detection_module.update_parameters(self.parameters)
        self.deep_learning_module.update_parameters(self.parameters)
        # self.graph_module.fetch_parameters()
        self.visualization_module.update_parameters(self.parameters)

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
        if self.output_data_item is None:
            descriptor = self.api.create_data_descriptor(is_sequence=False, collection_dimension_count=0,
                                                         datum_dimension_count=2)

            dummy_data = np.zeros((1, 1, 3), dtype=np.uint8)
            xdata = self.api.create_data_and_metadata(dummy_data, data_descriptor=descriptor)
            self.output_data_item = self.api.library.create_data_item_from_data_and_metadata(xdata)

        self.update_parameters()

        model = self.model

        print('start processing')
        with self.api.library.data_ref_for_data_item(self.output_data_item) as data_ref:

            def thread_this(stop_live_analysis_event, camera, data_ref):
                while not stop_live_analysis_event.is_set():
                    # if not camera.is_playing:
                    #    self.stop_live_analysis()

                    source_data = camera.grab_next_to_finish()  # TODO: This starts scanning? Must be a bug.

                    orig_images = source_data[0].data.copy()
                    points = model.predict(orig_images)

                    # self.output_data_item.title = 'Analysis of ' + camera.get_property_as_str('name')

                    visualization = create_visualization(orig_images, model.last_density, model.last_segmentation,
                                                         points, self.parameters['visualization'])
                    # visualization = self.visualization_module.add_points(visualization, points)

                    data_ref.data = visualization

            self.thread = threading.Thread(target=thread_this,
                                           args=(self.stop_live_analysis_event, self.get_camera(), data_ref))
            self.thread.start()


# class StructureRecognitionExtension(object):
#     extension_id = "nion.swift.extension.structure_recognition"
#
#     def __init__(self, api_broker):
#         api = api_broker.get_api(version='~1.0', ui_version='~1.0')
#         self.__panel_ref = api.create_panel(StructureRecognitionPanelDelegate(api))
#
#     def close(self):
#         self.__panel_ref.close()
#         self.__panel_ref = None

from nion.swift import Workspace

workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(StructureRecognitionPanel, "structure-recognition-panel", _("Structure Recognition"),
                                 ["left", "right"], "right")
