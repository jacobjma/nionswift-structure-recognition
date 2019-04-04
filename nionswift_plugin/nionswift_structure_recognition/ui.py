import gettext

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .graph import stable_delaunay_graph, faces_to_quadedge
from .utils import gaussian_filter, density2points, rescale, ensemble_expand, ensemble_reduce
from .utils import normalize_global, normalize_local, pad_to_closest_multiple, largest_connected_component_without_holes
from .visualise import add_points, add_line_segments, convert2color, get_colors_array
from .widgets import SectionWidget, ScrollAreaWidget

_ = gettext.gettext


class StructureRecognitionPanelDelegate(object):

    def __init__(self, api):
        self.__api = api
        self.panel_id = "structure-recognition-panel"
        self.panel_name = _("Structure Recognition")
        self.panel_positions = ["left", "right"]
        self.panel_position = "right"
        self._data = {}

        self.ui = None
        self.document_controller = None
        self.output_data_item = None
        self.target_data_item = None
        self.model = None

    def add_section(self, title):
        section = SectionWidget(self.ui, title)
        self.column.add(section)
        return section

    def load_model(self):
        json_file = open(self.architecture_widget.text, 'r')
        self.model = keras.models.model_from_json(json_file.read())
        json_file.close()
        self.model.load_weights(self.parameters_widget.text)

    def set_target_data_item(self):
        self.target_data_item = self.document_controller.target_data_item

        if self.target_data_item == self.output_data_item:
            raise RuntimeError()

    def get_target_image_data(self):
        if self.target_data_item is None:
            raise RuntimeError()
        return self.target_data_item.xdata.data

    def get_target_shape(self):
        if self.target_data_item.xdata.is_sequence:
            print("SEQUENCE")
            print(self.target_data_item.xdata.dimensional_shape[1:])
            return self.target_data_item.xdata.dimensional_shape[1:]
        else:
            print("IMAGE")
            print(self.target_data_item.xdata.dimensional_shape[:2])
            return self.target_data_item.xdata.dimensional_shape[:2]

    def get_target_image_data_slice(self):
        if self.target_data_item.xdata.is_sequence:
            return self.target_data_item._DataItem__display_item.display_data_channel.sequence_index
        else:
            return self.target_data_item._DataItem__display_item.display_data_channel.slice_center

    def new_output_dataitem(self):
        self.output_data_item = self.document_controller.library.create_data_item()

    def get_images(self, i=None):
        if i is None:
            i = self.get_target_image_data_slice()

        if self.target_data_item.xdata.is_sequence:
            images = tf.convert_to_tensor(self.get_target_image_data()[None, i, ..., None].copy(), dtype=tf.float32)
        else:
            images = tf.convert_to_tensor(self.get_target_image_data()[None, ..., i, None].copy(), dtype=tf.float32)
        print("SHAPE:")
        print(images.shape)
	

        if self.resample_widget.text is not '':
            scale_factor = float(self.resample_widget.text) / self.target_data_item.dimensional_calibrations[1].scale
        else:
            scale_factor = 1.

        if scale_factor != 1.:
            images = rescale(images, scale_factor)

        # print(images)

        images = pad_to_closest_multiple(images, 16)

        if self.normalization_widget.text == '':
            images = normalize_global(images)

        else:
            normalisation_sigma = float(self.normalization_widget.text)

            images = normalize_local(images, normalisation_sigma)

        # print(images)

        return images

    def get_predictions(self, images):
        assert images.shape[0] > 0

        if self.ensembling_widget.text == '':
            density, confidence = self.model.predict(images)

        else:
            ensemble_images = ensemble_expand(images.numpy()[0])
            density, confidence = self.model.predict(ensemble_images)
            density = ensemble_reduce(density)[None]
            confidence = ensemble_reduce(confidence)[None]

        # density = tf.image.resize(density, self.get_target_shape()[1:])
        # confidence = tf.image.resize(confidence, self.get_target_shape()[1:])
        density = tf.image.resize(density, self.get_target_shape()[:2])
        confidence = tf.image.resize(confidence, self.get_target_shape()[:2])

        confidence_smearing = float(self.confidence_smear_widget.text)
        confidence_threshold = float(self.confidence_threshold_widget.text)
        confidence_region = tf.cast(gaussian_filter(confidence, confidence_smearing) > confidence_threshold,
                                    tf.float32).numpy()

        # print(density, confidence)

        threshold = float(self.integration_threshold_widget.text)

        points = []
        for i in range(images.shape[0]):
            confidence_region[i, ..., 0] = largest_connected_component_without_holes(confidence_region[i, ..., 0])

            confident_density = density[i, ..., 0] * confidence_region[i, ..., 0]

            points.append(density2points(confident_density, confident_density > threshold))

        return density, confidence, confidence_region, points

    def create_output_images(self, images, density, confidence, confidence_region, points):
        background_map = self.background_map_widget.current_item

        images = tf.image.resize(images, self.get_target_shape()[:2])

        if background_map == 'Image':
            output_images = images[..., 0, None].numpy()

        elif background_map == 'Density':
            output_images = density[..., 0, None].numpy()

        elif background_map == 'Confidence':
            output_images = confidence[..., 0, None].numpy()

        elif background_map == 'Confidence Region':
            output_images = confidence_region[..., 0, None]

        else:
            raise RuntimeError()

        output_images = convert2color(output_images)

        for i in range(output_images.shape[0]):

            if self.overlay_graph_widget.checked | self.overlay_faces_widget.checked:
                if self.alpha_widget.text == '':
                    raise RuntimeError()
                else:
                    faces, face_sizes, edges = self.build_graph(points[i])

            if self.overlay_faces_widget.checked:
                colors = get_colors_array(face_sizes, cmap='tab10', vmin=3, vmax=13)[:, :3]
                colors = np.round(255 * colors)
                for face, face_size, color in zip(faces, face_sizes, colors):
                    outline = np.fliplr(np.round(points[i][face[:face_size]]).astype(int))
                    cv2.fillConvexPoly(output_images[i], outline, tuple(color))

            if self.overlay_graph_widget.checked:
                output_images[i] = add_line_segments(output_images[i], points[i][edges])

            if self.overlay_points_widget.checked:
                output_images[i] = add_points(output_images[i], points[i])

        return output_images

    def build_graph(self, points):
        alpha = float(self.alpha_widget.text)
        remove_hull_adjacent = self.remove_hull_adjacent_widget.checked
        faces, face_sizes = stable_delaunay_graph(points, alpha, remove_hull_adjacent=remove_hull_adjacent,
                                                  remove_outlier_faces=True)

        quadedge = faces_to_quadedge(faces, face_sizes)

        edges = [list(key) for key in quadedge.keys()]

        return faces, face_sizes, edges

    def update(self):
        images = self.get_images()
        density, confidence, confidence_region, points = self.get_predictions(images)

        output_images = self.create_output_images(images, density, confidence, confidence_region, points)

        self.output_data_item.set_data(output_images[0])

    def loop_series(self):

    def create_panel_widget(self, ui, document_controller):
        self.ui = ui
        self.document_controller = document_controller

        self.column = ui.create_column_widget()

        scroll_area = ScrollAreaWidget(ui._ui)
        scroll_area.content = self.column._widget

        push_button = self.ui.create_push_button_widget('Update')
        push_button.on_clicked = self.update
        self.column.add(push_button)

        #################### Model ####################

        model_section = self.add_section('Model')
        self.architecture_widget = model_section.add_line_edit('Architecture',
                                                               default_text='models/model.json')
        self.parameters_widget = model_section.add_line_edit('Parameters',
                                                             default_text='models/graphene-stem.h5')
        self.load_model_pushbutton = model_section.add_push_button('Load', self.load_model)

        self.resample_widget = model_section.add_line_edit('Resample', placeholder_text='No Resampling')
        self.normalization_widget = model_section.add_line_edit('Normalization Sigma', placeholder_text='Use Global')
        self.ensembling_widget = model_section.add_line_edit('Ensemble Size', placeholder_text='No Ensembling')
        self.integration_threshold_widget = model_section.add_line_edit('Integration Threshold', default_text='.5')

        #################### Graph ####################

        graph_section = SectionWidget(ui, 'Graph')
        self.column.add(graph_section)
        self.alpha_widget = graph_section.add_line_edit('Alpha', default_text=1.2, placeholder_text='No Graph')
        self.outlier_sigma_widget = graph_section.add_line_edit('Outlier Sigma', default_text=3,
                                                                placeholder_text='No Outlier Removal')
        self.remove_hull_adjacent_widget = graph_section.add_check_box('Remove Hull Adjacent')

        #################### Confidence ####################

        confidence_section = SectionWidget(ui, 'Confidence')
        self.column.add(confidence_section)
        self.confidence_method_widget = confidence_section.add_combo_box('Confidence Region', ['Predicted'])
        self.confidence_threshold_widget = confidence_section.add_line_edit('Confidence Threshold', .6)
        self.confidence_smear_widget = confidence_section.add_line_edit('Smearing', 20)
        self.confidence_connected_widget = confidence_section.add_check_box('Ensure Connected')

        #################### Visualisation ####################

        visualisation_section = SectionWidget(ui, 'Visualisation')
        self.column.add(visualisation_section)
        self.background_map_widget = visualisation_section.add_combo_box('Background Map', ['Image',
                                                                                            'Density',
                                                                                            'Confidence',
                                                                                            'Confidence Region'])
        self.overlay_points_widget = visualisation_section.add_check_box('Overlay Points')
        self.overlay_graph_widget = visualisation_section.add_check_box('Overlay Graph')
        self.overlay_faces_widget = visualisation_section.add_check_box('Overlay Faces')

        #################### Output ####################

        output_section = self.add_section('Output')
        output_section.add_push_button('Set Target Data Item', self.set_target_data_item)
        output_section.add_push_button('New Output Data Item', self.new_output_dataitem)


        #self.predict_series_widget = output_section.add_check_box('Predict Series')


        # self.predict_series_widget = output_section.add_check_box('Predict Series')

        # main_column.add(DeepLearningSection(ui, document_controller, self._data))
        # main_column.add(PointSegmentMatchingSection(ui, document_controller, self._data))
        self.column.add_stretch()

        return scroll_area


class StructureRecognitionExtension(object):
    # required for Swift to recognize this as an extension class.
    extension_id = "nion.swift.extension.structure_recognition"

    def __init__(self, api_broker):
        # grab the api object.
        print(api_broker)
        api = api_broker.get_api(version='~1.0', ui_version='~1.0')
        # be sure to keep a reference or it will be closed immediately.
        self.__panel_ref = api.create_panel(StructureRecognitionPanelDelegate(api))

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__panel_ref.close()
        self.__panel_ref = None
