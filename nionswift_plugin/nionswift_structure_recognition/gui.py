import gettext
import threading

import numpy as np
import torch
import torch.nn as nn

from .dl import DeepLearningModule, pad_to_size
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
        #return self.api.get_instrument_by_id("autostem_controller",version="1")
        return self.api.get_hardware_source_by_id("superscan",version="1")
        #return self.api.get_hardware_source_by_id("nion1010",version="1")
        #return self.api.get_hardware_source_by_id('usim_scan_device', '1.0')

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

        print('start processing')
        with self.api.library.data_ref_for_data_item(self.output_data_item) as data_ref:

            def thread_this(stop_live_analysis_event, camera, data_ref):
                while not stop_live_analysis_event.is_set():
                    if not camera.is_playing:
                        self.stop_live_analysis()

                    source_data = camera.grab_next_to_finish()  # TODO: This starts scanning? Must be a bug.

                    images = source_data[0].data.copy()
                    orig_shape = images.shape[-2:]

                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    images = torch.tensor(images).to(device)

                    sampling = self.scale_detection_module.detect_scale(images)
                    print('detected sampling:', sampling)
                    
                    images = self.deep_learning_module.reshape_images(images)
                    images = self.deep_learning_module.rescale_images(images, sampling)
                    images = self.deep_learning_module.normalize_images(images)
                    images = pad_to_size(images, images.shape[2], images.shape[3], n=16)

                    classes = nn.Softmax(1)(self.deep_learning_module.mask_model(images))
                    mask = torch.sum(classes[:, :-1], dim=1)[:, None]

                    images = self.deep_learning_module.normalize_images(images, mask)
                    density = nn.Sigmoid()(self.deep_learning_module.density_model(images))
                    
                    density = density * mask
                    density = density.detach().cpu().numpy()

                    points = self.deep_learning_module.nms(density)
                    points = self.deep_learning_module.postprocess_points(points, density.shape[2:], orig_shape,
                                                                          sampling)

                    density = self.deep_learning_module.postprocess_images(density[0, 0], orig_shape, sampling)

                    classes = classes[0].detach().cpu().numpy()
                    classes = np.argmax(classes, axis=0).astype(np.float)

                    classes = self.deep_learning_module.postprocess_images(classes, orig_shape, sampling)

                    #visualization = self.visualization_module.create_background(source_data[0].data, classes, density)
                    visualization = self.visualization_module.create_background(source_data[0].data, classes, density)
                    visualization = self.visualization_module.add_points(visualization, points)

                    data_ref.data = visualization
                    # data_ref.data = mask[0, 0].detach().cpu().numpy()
                    # print(i)
                    # i += 1

                #     preprocessed = self.deep_learning_module.preprocess_image(image, sampling)
                #
                #     density, classes = self.deep_learning_module.forward_pass(preprocessed)
                #
                #     density = density.detach().numpy()
                #     classes = classes.detach().numpy()
                #
                #     classes = merge_overlapping(classes, [2, 3])
                #     classes = merge_overlapping(classes, [3, 2])
                #
                #     mask = gaussian(classes[:, :-1].sum(axis=1), 10) ** 2
                #
                #     density = density * mask
                #

                #
                #     density = self.deep_learning_module.postprocess_image(density[0, 0], image.shape, sampling)
                #     classes = self.deep_learning_module.postprocess_image(
                #         np.argmax(classes[0], axis=0).astype(np.float),
                #         image.shape,
                #         sampling)
                #
                #     visualization = self.visualization_module.create_visualization(image, density, classes, graph, rmsd,
                #                                                                    probabilities)

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
