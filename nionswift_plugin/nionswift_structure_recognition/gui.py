import gettext
import threading

import numpy as np
import torch
import torch.nn as nn

from .dl import DeepLearningModule
# from .graph import GraphModule
from .scale import ScaleDetectionModule
from .visualization import VisualizationModule
from .widgets import ScrollArea, push_button_template, combo_box_template

_ = gettext.gettext


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

        #try:
        #    return self.api.get_hardware_source_by_id("superscan",version="1")
        # return self.api.get_hardware_source_by_id("nion1010",version="1")
        #except:
        #    pass

        if camera is None:
            return self.api.get_hardware_source_by_id('usim_scan_device', '1.0')
        else:
            return camera

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
                    # orig_shape = orig_images.shape[-2:]

                    points = model.predict(orig_images)['points'][0]

                    visualization = self.visualization_module.create_background(orig_images,
                                                                                model.last_density,
                                                                                model.last_segmentation
                                                                                )
                    visualization = self.visualization_module.add_points(visualization, points)
                    
                    def update_data_item():
                        data_ref.data = visualization
                    self.api.queue_task(update_data_item)

            camera = self.get_camera()

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
