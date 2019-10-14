import os

import numpy as np
import torch
from skimage.transform import rescale

from .nms import non_maximum_suppresion
from .unet import UNet, DensityMap, ClassificationMap
from .utils import StructureRecognitionModule
from .widgets import Section, line_edit_template


def closest_multiple_ceil(n, m):
    return int(np.ceil(n / m) * m)


def pad_to_size(image, height, width):
    up = (height - image.shape[0]) // 2
    down = height - image.shape[0] - up
    left = (width - image.shape[1]) // 2
    right = width - image.shape[1] - left
    image = np.pad(image, pad_width=((up, down), (left, right)), mode='constant', constant_values=((0., 0.), (0., 0.)))
    return image


def normalize_global(image):
    return (image - np.mean(image)) / np.std(image)


presets = {'graphene':
               {'weights_file': 'graphene.pt',
                'training_sampling': '0.005859375',
                'margin': '0.5',
                'nms_distance': '0.1',
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

        # model_row, self.model_line_edit = line_edit_template(self.ui, 'Model')
        parameters_row, self.parameters_line_edit = line_edit_template(self.ui, 'Weights')
        training_scale_row, self.training_sampling_line_edit = line_edit_template(self.ui, 'Training sampling [nm]')
        margin_row, self.margin_line_edit = line_edit_template(self.ui, 'Margin [nm]')
        nms_distance_row, self.nms_distance_line_edit = line_edit_template(self.ui, 'NMS distance [nm]')
        nms_threshold_row, self.nms_threshold_line_edit = line_edit_template(self.ui, 'NMS threshold')

        # section.column.add(model_row)
        section.column.add(parameters_row)
        section.column.add(training_scale_row)
        section.column.add(margin_row)
        section.column.add(nms_distance_row)
        section.column.add(nms_threshold_row)

    def set_preset(self, name):
        # self.model_line_edit.text = presets[name]['model_file']
        self.parameters_line_edit.text = presets[name]['weights_file']
        self.training_sampling_line_edit.text = presets[name]['training_sampling']
        self.margin_line_edit.text = presets[name]['margin']
        self.nms_distance_line_edit.text = presets[name]['nms_distance']
        self.nms_threshold_line_edit.text = presets[name]['nms_threshold']

    def preprocess_image(self, image, sampling):
        # if len(images.shape) == 2:
        #     images = np.expand_dims(np.expand_dims(images, 0), 0)
        #
        # elif len(images.shape) == 3:
        #     images = np.expand_dims(images, 0)
        #
        # elif len(images.shape) != 4:
        #     raise RuntimeError('')

        scale = sampling / self.training_sampling
        # print(scale)
        image = normalize_global(rescale(image, scale, multichannel=False, anti_aliasing=False))

        margin = self.margin / sampling

        new_height = closest_multiple_ceil(image.shape[0] + margin, 16)
        new_width = closest_multiple_ceil(image.shape[1] + margin, 16)

        image = pad_to_size(image, new_height, new_width)

        return torch.from_numpy(image[None, None].astype(np.float32))

    def postprocess_image(self, image, original_shape, sampling):
        image = rescale(image, self.training_sampling / sampling, multichannel=False, anti_aliasing=False)
        shape = image.shape
        padding = (shape[0] - original_shape[0], shape[1] - original_shape[1])
        image = image[padding[0] // 2: padding[0] // 2 + original_shape[0],
                padding[1] // 2: padding[1] // 2 + original_shape[1]]
        return image

    def postprocess_points(self, points, shape, original_shape, sampling):
        shape = np.round(np.array(shape) * self.training_sampling / sampling)
        padding = (shape[0] - original_shape[0], shape[1] - original_shape[1])
        points = points * self.training_sampling / sampling
        return points - np.array([padding[0] // 2, padding[1] // 2])

    def forward_pass(self, preprocessed_image):
        density, classes = self.model(preprocessed_image)
        return density, classes

    def nms(self, density, classes):
        nms_distance_pixels = int(np.round(self.nms_distance / self.training_sampling))

        accepted, probabilities = non_maximum_suppresion(density, classes, nms_distance_pixels, self.nms_threshold)

        points = np.array(np.where(accepted[0])).T
        probabilities = probabilities[0, :, points[:, 0], points[:, 1]]
        return points, probabilities

    def fetch_parameters(self):
        self.training_sampling = float(self.training_sampling_line_edit.text)
        self.margin = float(self.margin_line_edit.text)

        models_dir = os.path.join(os.path.dirname(__file__), 'models')

        # self.model_file = os.path.join(models_dir, self.model_line_edit.text)
        self.parameters_file = os.path.join(models_dir, self.parameters_line_edit.text)

        self.model = UNet([DensityMap(), ClassificationMap(4)], init_features=32, in_channels=1, p=0.)
        self.model.load_state_dict(torch.load(self.parameters_file, map_location=torch.device('cpu')))

        # json_file = open(self.model_file, 'r')
        # self.model = keras.models.model_from_json(json_file.read())
        # self.model.load_weights(self.parameters_file)

        self.nms_distance = float(self.nms_distance_line_edit.text)
        self.nms_threshold = float(self.nms_threshold_line_edit.text)
