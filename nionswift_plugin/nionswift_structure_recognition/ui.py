import gettext

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from nion.data import xdata_1_0 as xd
from nion.ui.Widgets import SectionWidget
from psm.build import regular_polygons
from psm.graph.graphutils import adjacency2edges
from psm.match import RMSD
from psm.segments import Segments
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label
from skimage.segmentation import clear_border
from skimage.transform import rescale
from tensorflow import keras

from .utils import ensemble_reduce, ensemble_expand, standardize_image

_ = gettext.gettext


class SectionWidgetWrapper:

    def __init__(self, ui, title):
        self.__ui = ui
        self.__section_content_column = self.__ui._ui.create_column_widget()
        self.__section_widget = SectionWidget(self.__ui._ui, title, self.__section_content_column, 'test')

    @property
    def _ui(self):
        return self.__ui

    def add(self, widget):
        self.__section_content_column.add(widget._widget)

    @property
    def _widget(self):
        return self.__section_widget


class StructureRecognitionSection(SectionWidgetWrapper):

    def __init__(self, ui, document_controller, title, data=None):
        super().__init__(ui, title)
        self._document_controller = document_controller
        if data is None:
            self._data = {}
        else:
            self._data = data

        self._column = ui.create_column_widget()
        self.add(self._column)
        self._widgets = {}
        self._left_margin = 0
        self._right_margin = 10

    def add_text_box(self, label, value, tag):
        row = self._ui.create_row_widget()
        row.add_spacing(self._left_margin)
        row.add(self._ui.create_label_widget(_(label)))
        row.add_spacing(5)
        combo_box = self._ui.create_line_edit_widget(value)
        row.add(combo_box)
        row.add_spacing(self._right_margin)
        self._widgets[tag] = combo_box
        self._column.add(row)

    def add_combo_box(self, label, items, tag):
        row = self._ui.create_row_widget()
        row.add_spacing(self._left_margin)
        row.add(self._ui.create_label_widget(_(label)))
        combo_box = self._ui.create_combo_box_widget(items=items)
        row.add(combo_box)
        row.add_spacing(self._right_margin)
        self._widgets[tag] = combo_box
        self._column.add(row)

    def add_push_button(self, label, callback):
        row = self._ui.create_row_widget()
        row.add_spacing(self._left_margin)
        push_button = self._ui.create_push_button_widget(_(label))
        push_button.on_clicked = callback
        row.add(push_button)
        row.add_spacing(self._right_margin)
        self._column.add(row)

    def add_check_box(self, label, tag):
        row = self._ui.create_row_widget()
        row.add_spacing(self._left_margin)
        check_box = self._ui.create_check_box_widget(label)
        row.add(check_box)
        row.add_spacing(self._right_margin)
        self._widgets[tag] = check_box
        self._column.add(row)

    def get_data_item(self):
        return self._document_controller.target_data_item

    def get_image_data(self):
        return self._document_controller.target_data_item.xdata.data

    def get_image_color(self):
        image = self.get_image_data()
        image = (image - image.min()) / (image.max() - image.min())
        return cv2.cvtColor(np.uint8(255 * image).T, cv2.COLOR_GRAY2RGB)


class DeepLearningSection(StructureRecognitionSection):

    def __init__(self, ui, document_controller, data):
        super().__init__(ui, document_controller, title='Deep Learning', data=data)

        self._model = None

        self._graph = None

        self.add_text_box('Model', 'models/model.json', tag='model')

        self.add_text_box('Weights', 'models/weights.h5', tag='weights')

        self.add_text_box('Training Scale [nm]', '0.0039', tag='scale')

        self.add_push_button('Load', self.load_model)

        self.add_combo_box('Classes', items=['No model loaded'], tag='class')

        self.add_check_box('Use ensemble', tag='ensemble')

        self.add_text_box('Clear Border', 20, tag='border')

        self.add_push_button('Detect Structures', self.show_detected)

        self.add_check_box('Create Point Regions', tag='create_point_regions')

        self.add_push_button('Show Density', self.show_density)

    def load_model(self):
        json_file = open(self._widgets['model'].text, 'r')
        self._model = keras.models.model_from_json(json_file.read())
        json_file.close()
        self._model.load_weights(self._widgets['weights'].text)
        self._graph = tf.get_default_graph()

        outdim = self._model.layers[-1].output_shape[-1]
        self._widgets['class'].items = ['All'] + ['Class # {}'.format(i) for i in range(outdim - 1)]

    def get_density(self):

        if self._model is None:
            raise RuntimeError('Set a recognition model')

        image = self.get_image_data()

        old_shape = image.shape

        dataitem = self._document_controller.target_data_item

        target_scale = float(self._widgets['scale'].text)

        scale = dataitem.dimensional_calibrations[0].scale

        image = rescale(image, scale / target_scale)

        shape = (np.ceil(np.array(image.shape) / 16) * 16).astype(int)

        transformed_image = np.zeros(shape)

        transformed_image[:image.shape[0], :image.shape[1]] = image

        transformed_image = standardize_image(transformed_image)

        if self._widgets['ensemble'].checked:
            transformed_image = ensemble_expand(transformed_image)[..., None]
        else:
            transformed_image = transformed_image[None, ..., None]

        with self._graph.as_default():
            prediction = self._model.predict(transformed_image)

            if self._widgets['ensemble'].checked:
                prediction = ensemble_reduce(prediction)
            else:
                prediction = prediction[0]

            return rescale(prediction, target_scale / scale)[:old_shape[0], :old_shape[1]]

    def show_density(self):
        density = self.get_density()

        dataitem = self._document_controller.library.create_data_item()

        selected_class = self._widgets['class'].current_index

        if selected_class == 0:
            prediction = 1 - density[..., -1]
        else:
            prediction = density[..., selected_class - 1]

        dataitem.set_data(prediction)

    def detect(self):
        density = self.get_density()

        thresholded = density[..., -1] < 1 - .2
        thresholded = clear_border(thresholded, int(self._widgets['border'].text))
        label_image, num_labels = label(thresholded, return_num=True)

        class_probabilities = np.zeros((num_labels - 1, density.shape[-1] - 1))

        for label_num in range(1, num_labels):
            class_totals = np.sum(density[label_image == label_num, :-1], axis=0)
            class_probabilities[label_num - 1] = class_totals / np.sum(class_totals)

        centers = np.array(center_of_mass(1 - density[..., -1], label_image, range(1, num_labels)))
        class_ids = np.argmax(class_probabilities, axis=1)
        selected_class = self._widgets['class'].current_index
        if selected_class != 0:
            centers = centers[class_ids == selected_class - 1]
            class_ids = class_ids[class_ids == selected_class - 1]

        self._data['centers'] = centers
        self._data['class_ids'] = class_ids

    def show_detected(self):
        self.detect()

        if self._widgets['create_point_regions'].checked:
            shape = self.get_data_item().xdata.data_shape

            dataitem = self._document_controller.create_data_item_from_data_and_metadata(self.get_data_item().xdata,
                                                                                         title='Local Maxima of ' +
                                                                                               self.get_data_item().title)

            for center in self._data['centers']:
                dataitem.add_point_region(center[0] / shape[0], center[1] / shape[1])

        # self.detect()
        #
        #
        #
        # centers_rounded = np.round(self._data['centers']).astype(np.int)
        # colors = {0: (255, 0, 0), 1: (0, 255, 0)}
        # size = 10
        #
        # image = self.get_image_color()
        #
        # for center, class_id in zip(centers_rounded, self._data['class_ids']):
        #     cv2.circle(image, tuple(center), size, colors[class_id], 2)
        #     # cv2.circle(image, tuple(center), 3, colors[class_id], 10)
        #     # cv2.rectangle(image, tuple(center - size), tuple(center + size), colors[class_id], 1)
        #
        # xdata = xd.rgb(image[..., 0], image[..., 1], image[..., 2])
        #
        # self._document_controller.create_data_item_from_data_and_metadata(xdata, title='')


class PointSegmentMatchingSection(StructureRecognitionSection):

    def __init__(self, ui, document_controller, data):
        super().__init__(ui, document_controller, title='Point Segment Matching', data=data)

        self.add_text_box('alpha', '1.2', tag='alpha')

        self.add_push_button('Build Graph', self.show_graph)

        self.add_combo_box('Segments', items=['Faces', 'Traversals'], tag='segments')

        self.add_combo_box('Templates', items=['Polygons', 'Structure', 'Learn'], tag='templates')

        self.add_push_button('Register Segments', self.register_segments)

        self.add_combo_box('CMap', items=['viridis', 'hsv', 'seismic', 'rainbow'], tag='cmap')

        self.add_combo_box('Measure', items=['RMSD', 'Match', 'exx', 'eyy', 'eyz', 'Rotation'], tag='show')

        self.add_push_button('Show', self.show)

    def build_graph(self):
        segments = Segments(self._data['centers'])
        segments.build_graph(float(self._widgets['alpha'].text))

        self._data['segments'] = segments

    def show_graph(self):
        self.build_graph()
        centers_rounded = np.round(self._data['centers']).astype(np.int)
        image = self.get_image_color()

        for edge in adjacency2edges(self._data['segments'].adjacency):
            cv2.line(image, tuple(centers_rounded[edge[0]]), tuple(centers_rounded[edge[1]]), color=(255, 0, 0),
                     thickness=2)

        xdata = xd.rgb(image[..., 0], image[..., 1], image[..., 2])
        self._document_controller.create_data_item_from_data_and_metadata(xdata, title='')

    def register_segments(self):
        templates = regular_polygons(1.4, [5, 6, 7])

        rmsd_calc = RMSD(transform='similarity', pivot='cop')

        self._data['segments'].faces(remove_hull=True)

        self._data['segments'].register(templates, rmsd_calc=rmsd_calc, progress_bar=False)

        best_match, best_rmsd = self._data['segments'].best_matches()
        strain, rotation = rmsd_calc.calc_strain(self._data['segments'])

        self._data['rmsd'] = best_rmsd
        self._data['match'] = best_match
        self._data['exx'] = strain[:, 0, 0]
        self._data['eyy'] = strain[:, 1, 1]
        self._data['exy'] = strain[:, 1, 0]
        self._data['rotation'] = rotation % (2 * np.pi / 6)

    def show(self):
        data = self._data[self._widgets['show'].current_item.lower()]

        data = ((data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data)) * 255).astype(int)

        cmap = matplotlib.cm.get_cmap(self._widgets['cmap'].current_item)

        image = self.get_image_color()

        for i, segment_indices in enumerate(self._data['segments'].indices):
            outline = np.round(self._data['centers'][segment_indices]).astype(int)

            if not np.isnan(data[i]):
                color = tuple(int(r * 255) for r in cmap(data[i])[:-1])
            else:
                color = (127, 127, 127)

            cv2.fillConvexPoly(image, outline, color)

        centers_rounded = np.round(self._data['centers']).astype(np.int)
        for edge in adjacency2edges(self._data['segments'].adjacency):
            cv2.line(image, tuple(centers_rounded[edge[0]]), tuple(centers_rounded[edge[1]]), color=(0, 0, 0),
                     thickness=2)

        xdata = xd.rgb(image[..., 0], image[..., 1], image[..., 2])

        self._document_controller.create_data_item_from_data_and_metadata(xdata, title='')


class StructureRecognitionPanelDelegate(object):

    def __init__(self, api):
        self.__api = api
        self.panel_id = "structure-recognition-panel"
        self.panel_name = _("Structure Recognition")
        self.panel_positions = ["left", "right"]
        self.panel_position = "right"
        self._data = {}

    def create_panel_widget(self, ui, document_controller):
        self.ui = ui
        self.document_controller = document_controller

        main_column = ui.create_column_widget()
        main_column.add_spacing(5)
        main_column.add(DeepLearningSection(ui, document_controller, self._data))
        main_column.add(PointSegmentMatchingSection(ui, document_controller, self._data))
        main_column.add_stretch()

        return main_column


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
