import gettext
import threading

import numpy as np

from .model import presets, build_model_from_dict
from .visualization import VisualizationModule
from .widgets import ScrollArea, push_button_template, combo_box_template, StructureRecognitionModule, \
    line_edit_template, Section

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

    def fetch_parameters(self):
        self.crystal_system = self.crystal_system_combo_box._widget.current_item.lower()
        self.lattice_constant = float(self.lattice_constant_line_edit._widget.text)
        self.min_sampling = float(self.min_sampling_line_edit._widget.text)


class DeepLearningModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

        self.training_sampling = None
        self.mask_model = None
        self.density_model = None
        self.nms_distance = None

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
        self.mask_weights_line_edit.text = presets[name]['mask_model']['weights']
        self.density_weights_line_edit.text = presets[name]['density_model']['weights']
        self.training_sampling_line_edit.text = presets[name]['training_sampling']
        self.margin_line_edit.text = presets[name]['margin']
        self.nms_distance_line_edit.text = presets[name]['nms']['distance']
        self.nms_threshold_line_edit.text = presets[name]['nms']['threshold']

    def forward_pass(self, preprocessed_image):
        density, classes = self.model(preprocessed_image)
        return density, classes

    def fetch_parameters(self):
        self.training_sampling = float(self.training_sampling_line_edit.text)
        self.margin = float(self.margin_line_edit.text)
        self.nms_distance = float(self.nms_distance_line_edit.text)
        self.nms_threshold = float(self.nms_threshold_line_edit.text)
        self.model = build_model_from_dict(presets['graphene'])


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

    def create_panel_widget(self, ui, document_controller):
        self.ui = ui
        self.document_controller = document_controller
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

        # stop_row, stop_push_button = push_button_template(ui, 'Stop', stop_live_analysis)

        live_analysis_row = ui.create_row_widget()
        live_analysis_row.add(run_row)
        # live_analysis_row.add(stop_row)
        main_column.add(live_analysis_row)
        #
        self.scale_detection_module.create_widgets(main_column)
        self.deep_learning_module.create_widgets(main_column)
        # self.graph_module.create_widgets(main_column)
        self.visualization_module.create_widgets(main_column)

        self.preset_combo_box.current_item = 'Graphene'
        main_column.add_stretch()

        return scroll_area

    def process_sequence(self):
        pass
        # if source_data_item.xdata.is_sequence:
        #     current_index = source_data_item._DataItem__display_item.display_data_channel.sequence_index
        #     data = data[current_index]

    def get_camera(self):
        # return self.api.get_instrument_by_id("autostem_controller",version="1")

        camera = self.api.get_hardware_source_by_id("superscan", version="1")
        if camera is None:
            camera = self.api.get_hardware_source_by_id('usim_scan_device', '1.0')

        return camera
        # return self.api.get_hardware_source_by_id("nion1010",version="1")
        # return self.api.get_hardware_source_by_id('usim_scan_device', '1.0')

    def update_parameters(self):
        self.scale_detection_module.fetch_parameters()
        self.deep_learning_module.fetch_parameters()
        # self.graph_module.fetch_parameters()
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
        # def on_first_frame():
        # self.output_data_item = self.document_controller.library.create_data_item()
        # self.output_data_item.title = 'Visualization'

        if self.output_data_item is None:
            descriptor = self.api.create_data_descriptor(is_sequence=False, collection_dimension_count=0,
                                                         datum_dimension_count=2)

            dummy_data = np.zeros((1, 1, 3), dtype=np.uint8)
            xdata = self.api.create_data_and_metadata(dummy_data, data_descriptor=descriptor)
            self.output_data_item = self.api.library.create_data_item_from_data_and_metadata(xdata)

        self.update_parameters()

        model = self.deep_learning_module.model

        print('start processing')
        with self.api.library.data_ref_for_data_item(self.output_data_item) as data_ref:

            def thread_this(stop_live_analysis_event, camera, data_ref):
                while not stop_live_analysis_event.is_set():
                    if not camera.is_playing:
                        self.stop_live_analysis()

                    source_data = camera.grab_next_to_finish()  # TODO: This starts scanning? Must be a bug.

                    orig_images = source_data[0].data.copy()
                    points = model.predict(orig_images)

                    #self.output_data_item.title = 'Analysis of ' + camera.get_property_as_str('name')

                    visualization = self.visualization_module.create_background(orig_images,
                                                                                model.last_density,
                                                                                model.last_segmentation
                                                                                )
                    visualization = self.visualization_module.add_points(visualization, points)

                    data_ref.data = visualization

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
