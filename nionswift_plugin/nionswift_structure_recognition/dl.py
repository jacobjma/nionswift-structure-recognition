import os

import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import rescale

from .unet import UNet
from .utils import StructureRecognitionModule
from .widgets import Section, line_edit_template

presets = {'graphene':
               {'mask_weights_file': 'graphene_mask.pt',
                'density_weights_file': 'graphene_density.pt',
                'training_sampling': '0.05859375',
                'margin': '2',
                'nms_distance': '1',
                'nms_threshold': '0.5',
                }
           }


def closest_multiple_ceil(n, m):
    return int(np.ceil(n / m) * m)


def pad_to_size(images, height, width, n=None):
    if n is not None:
        height = closest_multiple_ceil(height, n)
        width = closest_multiple_ceil(width, n)

    shape = images.shape[-2:]

    up = (height - shape[0]) // 2
    down = height - shape[0] - up
    left = (width - shape[1]) // 2
    right = width - shape[1] - left
    images = F.pad(images, pad=[up, down, left, right])
    return images


def normalize_global(images):
    return (images - torch.mean(images, dim=(-2, -1), keepdim=True)) / torch.std(images, dim=(-2, -1), keepdim=True)


def weighted_normalization(images, mask=None):
    if mask is None:
        return normalize_global(images)

    weighted_means = torch.sum(images * mask, dim=(-1, -2), keepdim=True) / torch.sum(mask, dim=(-1, -2), keepdim=True)
    weighted_stds = torch.sqrt(
        torch.sum(mask * (images - weighted_means) ** 2, dim=(-1, -2), keepdim=True) /
        torch.sum(mask, dim=(-1, -2), keepdim=True))
    return (images - weighted_means) / weighted_stds


def sub2ind(rows, cols, array_shape):
    return rows * array_shape[1] + cols


def ind2sub(array_shape, ind):
    rows = ind // array_shape[1]
    cols = ind % array_shape[1]
    return (rows, cols)


def non_maximum_suppresion(density, distance, threshold, classes=None):
    shape = density.shape[2:]

    density = density.reshape((density.shape[0], -1))

    if classes is not None:
        classes = classes.reshape(classes.shape[:2] + (-1,))
        probabilities = np.zeros(classes.shape, dtype=classes.dtype)

    accepted = np.zeros(density.shape, dtype=np.bool_)
    suppressed = np.zeros(density.shape, dtype=np.bool_)

    x_disc = np.zeros((2 * distance + 1, 2 * distance + 1), dtype=np.int32)

    x_disc[:] = np.linspace(0, 2 * distance, 2 * distance + 1)
    y_disc = x_disc.copy().T
    x_disc -= distance
    y_disc -= distance
    x_disc = x_disc.ravel()
    y_disc = y_disc.ravel()

    r2 = x_disc ** 2 + y_disc ** 2

    x_disc = x_disc[r2 < distance ** 2]
    y_disc = y_disc[r2 < distance ** 2]

    weights = np.exp(-r2 / (2 * (distance / 3) ** 2))
    weights = np.reshape(weights[r2 < distance ** 2], (-1, 1))

    for i in range(density.shape[0]):
        suppressed[i][density[i] < threshold] = True
        for j in np.argsort(-density[i].ravel()):
            if not suppressed[i, j]:
                accepted[i, j] = True

                x, y = ind2sub(shape, j)
                neighbors_x = x + x_disc
                neighbors_y = y + y_disc

                valid = ((neighbors_x > -1) & (neighbors_y > -1) & (neighbors_x < shape[0]) & (
                        neighbors_y < shape[1]))

                neighbors_x = neighbors_x[valid]
                neighbors_y = neighbors_y[valid]

                k = sub2ind(neighbors_x, neighbors_y, shape)
                suppressed[i][k] = True

                if classes is not None:
                    tmp = np.sum(classes[i, :, k] * weights[valid], axis=0)
                    probabilities[i, :, j] = tmp / np.sum(tmp)

    accepted = accepted.reshape((density.shape[0],) + shape)

    if classes is not None:
        probabilities = probabilities.reshape(classes.shape[:2] + shape)
        return accepted, probabilities

    else:
        return accepted


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
