import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .utils import StructureRecognitionModule
from .widgets import Section, line_edit_template
from .nms import non_maximum_suppresion


def pad_to_closest_multiple(images, m):
    target_height = int(np.ceil(images.shape[1] / m) * m)
    target_width = int(np.ceil(images.shape[2] / m) * m)
    images = tf.image.pad_to_bounding_box(images, 0, 0, target_height, target_width)
    return images


def normalize_global(images):
    moments = tf.nn.moments(images, axes=[1, 2], keepdims=True)
    return (images - moments[0]) / tf.sqrt(moments[1])


def gaussian_kernel(sigma, truncate=None, dtype=tf.float32):
    if truncate is None:
        truncate = tf.math.ceil(3 * tf.cast(sigma, dtype))
    x = tf.cast(tf.range(-truncate, truncate + 1), dtype)
    return 1. / (np.sqrt(2. * np.pi) * sigma) * tf.exp(-x ** 2 / (2 * sigma ** 2))


def gaussian_filter(image, sigma, truncate=None):
    kernel = gaussian_kernel(sigma, truncate)
    image = tf.nn.conv2d(image, kernel[..., None, None, None], strides=[1, 1, 1, 1], padding='SAME')
    image = tf.nn.conv2d(image, kernel[None, ..., None, None], strides=[1, 1, 1, 1], padding='SAME')
    return image


def normalize_local(images, sigma):
    truncate = tf.cast(tf.math.ceil(3 * tf.cast(sigma, tf.float32)), tf.int32)
    images = tf.pad(images, [[0, 0], [truncate, truncate], [truncate, truncate], [0, 0]], 'REFLECT')
    mean = gaussian_filter(images, sigma, truncate)
    images = images - mean
    images = images / tf.sqrt(gaussian_filter(images ** 2, sigma, truncate))
    return images[:, truncate:-truncate, truncate:-truncate, :]


def rescale(images, scale):
    new_shape = np.round(np.array(images.shape[1:-1]) * scale).astype(int)
    return tf.image.resize(images, new_shape)


presets = {'graphene':
               {'model_file': 'u-net.json',
                'weights_file': 'graphene-stem.h5',
                'training_sampling': '0.0075',
                'margin': '0.5',
                'nms_distance': '0.05',
                'nms_threshold': '0.5',
                }
           }


class DeepLearningModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

        self.training_sampling = None

    def create_widgets(self, column):
        section = Section(self.ui, 'Deep learning')
        column.add(section)

        model_row, self.model_line_edit = line_edit_template(self.ui, 'Model')
        parameters_row, self.parameters_line_edit = line_edit_template(self.ui, 'Weights')
        training_scale_row, self.training_sampling_line_edit = line_edit_template(self.ui, 'Training sampling [nm]')
        margin_row, self.margin_line_edit = line_edit_template(self.ui, 'Margin [nm]')
        nms_distance_row, self.nms_distance_line_edit = line_edit_template(self.ui, 'NMS distance [nm]')
        nms_threshold_row, self.nms_threshold_line_edit = line_edit_template(self.ui, 'NMS threshold')

        section.column.add(model_row)
        section.column.add(parameters_row)
        section.column.add(training_scale_row)
        section.column.add(margin_row)
        section.column.add(nms_distance_row)
        section.column.add(nms_threshold_row)

    def set_preset(self, name):
        self.model_line_edit.text = presets[name]['model_file']
        self.parameters_line_edit.text = presets[name]['weights_file']
        self.training_sampling_line_edit.text = presets[name]['training_sampling']
        self.margin_line_edit.text = presets[name]['margin']
        self.nms_distance_line_edit.text = presets[name]['nms_distance']
        self.nms_threshold_line_edit.text = presets[name]['nms_threshold']

    def preprocess(self, images, sampling):
        images = tf.constant(images, dtype=tf.float32)

        if len(images.shape) == 2:
            images = tf.expand_dims(tf.expand_dims(images, -1), 0)

        elif len(images.shape) == 3:
            images = tf.expand_dims(images, -1)

        elif len(images.shape) != 4:
            raise RuntimeError('')

        scale = sampling / self.training_sampling

        images = normalize_global(rescale(images, scale))

        margin_pixels = self.margin / sampling

        input_shape = (int(np.ceil((images.shape[1] + 2 * margin_pixels) / 16) * 16),
                       int(np.ceil((images.shape[2] + 2 * margin_pixels) / 16) * 16))

        padding = (input_shape[0] - images.shape[1], input_shape[1] - images.shape[2])

        shift = (padding[0] // 2, padding[1] // 2)

        images = tf.image.pad_to_bounding_box(images, shift[0], shift[1], input_shape[0], input_shape[1])

        valid = (shift[0], (input_shape[0] - padding[0] // 2), shift[1], (input_shape[1] - padding[1] // 2))

        return images, valid, np.array(shift) * self.training_sampling

    def forward_pass(self, preprocessed_image):
        density, confidence = self.model(preprocessed_image)
        return density, confidence

    def nms(self, density):
        nms_distance_pixels = int(np.round(self.nms_distance / self.training_sampling))

        accepted = non_maximum_suppresion(density, nms_distance_pixels, self.nms_threshold)

        points = np.array(np.where(accepted)).T

        return points * self.training_sampling

    def fetch_parameters(self):

        self.training_sampling = float(self.training_sampling_line_edit.text)
        self.margin = float(self.margin_line_edit.text)

        models_dir = os.path.join(os.path.dirname(__file__), 'models')

        self.model_file = os.path.join(models_dir, self.model_line_edit.text)
        self.parameters_file = os.path.join(models_dir, self.parameters_line_edit.text)

        json_file = open(self.model_file, 'r')
        self.model = keras.models.model_from_json(json_file.read())
        self.model.load_weights(self.parameters_file)

        self.nms_distance = float(self.nms_distance_line_edit.text)
        self.nms_threshold = float(self.nms_threshold_line_edit.text)
