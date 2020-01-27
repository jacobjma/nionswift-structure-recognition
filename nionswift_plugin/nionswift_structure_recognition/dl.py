import os

import numpy as np
import torch
import torch.nn.functional as F
from abtem.learn.postprocess import non_maximum_suppresion
from abtem.learn.preprocess import weighted_normalization, pad_to_size
from abtem.learn.unet import UNet

from .utils import StructureRecognitionModule
from .widgets import Section, line_edit_template
from skimage.transform import rescale

presets = {'graphene':
               {'mask_weights_file': 'graphene_mask.pt',
                'density_weights_file': 'graphene_density.pt',
                'training_sampling': '0.05859375',
                'margin': '2',
                'nms_distance': '1',
                'nms_threshold': '0.5',
                }
           }


class DeepLearningModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

        self.training_sampling = None
        self.mask_model = None
        self.density_model = None

    def create_widgets(self, column):
        section = Section(self.ui, 'Deep learning')
        column.add(section)

        # model_row, self.model_line_edit = line_edit_template(self.ui, 'Model')
        mask_weights_row, self.mask_weights_line_edit = line_edit_template(self.ui, 'Mask weights')
        density_weights_row, self.density_weights_line_edit = line_edit_template(self.ui, 'Density weights')
        training_scale_row, self.training_sampling_line_edit = line_edit_template(self.ui, 'Training sampling [nm]')
        margin_row, self.margin_line_edit = line_edit_template(self.ui, 'Margin [nm]')
        nms_distance_row, self.nms_distance_line_edit = line_edit_template(self.ui, 'NMS distance [nm]')
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
        self.mask_weights_line_edit.text = presets[name]['mask_weights_file']
        self.density_weights_line_edit.text = presets[name]['density_weights_file']
        self.training_sampling_line_edit.text = presets[name]['training_sampling']
        self.margin_line_edit.text = presets[name]['margin']
        self.nms_distance_line_edit.text = presets[name]['nms_distance']
        self.nms_threshold_line_edit.text = presets[name]['nms_threshold']

    def reshape_images(self, images):
        if len(images.shape) == 2:
            images = images.unsqueeze(0).unsqueeze(0)

        elif len(images.shape) == 3:
            images = torch.unsqueeze(images, 0)

        elif len(images.shape) != 4:
            raise RuntimeError('')

        return images

    def rescale_images(self, images, sampling):
        scale_factor = sampling / self.training_sampling
        images = F.interpolate(images, scale_factor=scale_factor, mode='nearest')
        images = pad_to_size(images, images.shape[2], images.shape[3], n=16)
        return images

    def normalize_images(self, images, mask=None):
        return weighted_normalization(images, mask)

    def postprocess_images(self, image, original_shape, sampling):
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

    def load_model(self):
        models_dir = os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'), 'models')
        density_weights = os.path.join(models_dir, self.density_weights_line_edit.text)
        mask_weights = os.path.join(models_dir, self.mask_weights_line_edit.text)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.mask_model = UNet(in_channels=1, out_channels=3, init_features=32, dropout=0.)
        self.mask_model.load_state_dict(torch.load(mask_weights, map_location=torch.device('cpu')))
        self.mask_model.to(device)

        self.density_model = UNet(in_channels=1, out_channels=1, init_features=32, dropout=0.)
        self.density_model.load_state_dict(torch.load(density_weights, map_location=torch.device('cpu')))
        self.density_model.to(device)

    def forward_pass(self, preprocessed_image):
        density, classes = self.model(preprocessed_image)
        return density, classes

    def nms(self, density, classes=None):
        nms_distance_pixels = int(np.round(self.nms_distance / self.training_sampling))

        accepted = non_maximum_suppresion(density, distance=nms_distance_pixels,
                                          threshold=self.nms_threshold, classes=classes)

        points = np.array(np.where(accepted[0])).T
        # probabilities = probabilities[0, :, points[:, 0], points[:, 1]]
        return points  # , probabilities

    def fetch_parameters(self):
        self.training_sampling = float(self.training_sampling_line_edit.text)
        self.margin = float(self.margin_line_edit.text)
        self.nms_distance = float(self.nms_distance_line_edit.text)
        self.nms_threshold = float(self.nms_threshold_line_edit.text)

        self.load_model()

        # models_dir = os.path.join(os.path.dirname(__file__), 'models')
        #
        # # self.model_file = os.path.join(models_dir, self.model_line_edit.text)
        # self.parameters_file = os.path.join(models_dir, self.parameters_line_edit.text)
        #
        # self.model = UNet([DensityMap(), ClassificationMap(4)], init_features=32, in_channels=1, p=0.)
        # self.model.load_state_dict(torch.load(self.parameters_file, map_location=torch.device('cpu')))

        # json_file = open(self.model_file, 'r')
        # self.model = keras.models.model_from_json(json_file.read())
        # self.model.load_weights(self.parameters_file)
