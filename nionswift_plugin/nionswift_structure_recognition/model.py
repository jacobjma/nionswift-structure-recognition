import json
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom

from .scale import find_hexagonal_sampling
from .unet import UNet


def load_presets():
    presets_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'presets')

    presets = {}
    for file in os.listdir(presets_dir):
        with open(os.path.join(presets_dir, file)) as f:
            new_preset = json.load(f)
            presets[new_preset['name']] = new_preset
    return presets


presets = load_presets()


def sub2ind(rows, cols, array_shape):
    return rows * array_shape[1] + cols


def ind2sub(array_shape, ind):
    rows = (np.int32(ind) // array_shape[1])
    cols = (np.int32(ind) % array_shape[1])
    return (rows, cols)


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


def build_unet_model(weights_file):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    weights_file = os.path.join(os.path.join(os.path.dirname(__file__), 'models'), weights_file)
    weights = torch.load(weights_file, map_location=device)

    weights_list = list(weights.values())

    init_features = weights_list[0].shape[0]
    in_channels = weights_list[0].shape[1]
    out_channels = len(weights_list[-1])

    model = UNet(in_channels=in_channels,
                 out_channels=out_channels,
                 init_features=init_features,
                 dropout=0.)

    model.load_state_dict(weights)
    model.to(device)

    if out_channels == 1:
        return lambda x: nn.Sigmoid()(model(x))
    else:
        return lambda x: nn.Softmax(1)(model(x))


def build_model_from_dict(parameters):
    mask_model = build_unet_model(parameters['deep_learning']['mask_model'])
    density_model = build_unet_model(parameters['deep_learning']['density_model'])

    if parameters['scale']['crystal_system'] == 'hexagonal':
        scale_model = lambda x: find_hexagonal_sampling(x, lattice_constant=parameters['scale']['lattice_constant'],
                                                        min_sampling=parameters['scale']['min_sampling'])
    else:
        raise NotImplementedError('')

    def preprocess(images):
        pass

    def discretization_model(density, classes):
        nms_distance_pixels = int(
            np.round(parameters['nms']['distance'] / parameters['deep_learning']['training_sampling']))

        accepted, probabilities = non_maximum_suppresion(density, distance=nms_distance_pixels,
                                                         threshold=parameters['nms']['threshold'], classes=classes)

        points = [np.array(np.where(accepted[i])).T for i in range(accepted.shape[0])]
        probabilities = [probabilities[0, :, p[:, 0], p[:, 1]] for p in points]
        return points, probabilities

    model = AtomRecognitionModel(mask_model, density_model,
                                 training_sampling=parameters['deep_learning']['training_sampling'],
                                 margin=parameters['deep_learning']['training_sampling'],
                                 scale_model=scale_model, discretization_model=discretization_model)

    return model


class BatchGenerator:

    def __init__(self, n_items, max_batch_size):
        self._n_items = n_items
        self._n_batches = (n_items + (-n_items % max_batch_size)) // max_batch_size
        self._batch_size = (n_items + (-n_items % self.n_batches)) // self.n_batches

    @property
    def n_batches(self):
        return self._n_batches

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def n_items(self):
        return self._n_items

    def generate(self):
        batch_start = 0
        for i in range(self.n_batches):
            batch_end = batch_start + self.batch_size
            if i == self.n_batches - 1:
                yield batch_start, self.n_items - batch_end + self.batch_size
            else:
                yield batch_start, self.batch_size

            batch_start = batch_end


class AtomRecognitionModel:

    def __init__(self, mask_model, density_model, training_sampling, margin, scale_model, discretization_model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.preprocessing = preprocessing
        self.mask_model = mask_model
        self.density_model = density_model
        self.training_sampling = training_sampling
        self.margin = margin
        self.scale_model = scale_model
        self.discretization_model = discretization_model
        self.last_density = None
        self.last_segmentation = None

    def standardize_dims(self, images):
        if len(images.shape) == 2:
            images = images.unsqueeze(0).unsqueeze(0)
        elif len(images.shape) == 3:
            images = torch.unsqueeze(images, 1)
        elif len(images.shape) != 4:
            raise RuntimeError('')
        return images

    def rescale_images(self, images, sampling):
        scale_factor = sampling / self.training_sampling
        images = F.interpolate(images, scale_factor=scale_factor, mode='nearest')

        return images

    def normalize_images(self, images, mask=None):
        return weighted_normalization(images, mask)

    def postprocess_images(self, image, original_shape, sampling):
        image = zoom(image, self.training_sampling / sampling)
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

    def predict_batches(self, images, max_batch=4):
        images = torch.tensor(images).to(self.device)
        images = self.standardize_dims(images)
        batch_generator = BatchGenerator(len(images), max_batch)

        output = {'points': [], 'probabilities': []}
        for i, (start, size) in enumerate(batch_generator.generate()):
            print('Mini batch: {} of {}'.format(i, batch_generator.n_batches))
            new_output = self.predict(images[start:start + size])
            output['points'] += new_output['points']
            output['probabilities'] += new_output['probabilities']

        return output

    def predict(self, images):
        sampling = self.scale_model(images)
        images = torch.tensor(images).to(self.device)
        images = self.standardize_dims(images)
        orig_shape = images.shape[-2:]
        images = self.rescale_images(images, sampling)
        images = self.normalize_images(images)
        images = pad_to_size(images, images.shape[2], images.shape[3], n=16)
        segmentation = self.mask_model(images)
        mask = torch.sum(segmentation[:, :-1], dim=1)[:, None]

        images = self.normalize_images(images, mask)

        density = self.density_model(images)
        density = mask * density
        density = density.detach().cpu().numpy()

        classes = segmentation[:, :-1].detach().cpu().numpy()

        points, probabilities = self.discretization_model(density, classes)
        points = [self.postprocess_points(p, density.shape[-2:], orig_shape, sampling)[:, ::-1] for p in points]
        # points = [self.postprocess_points(p, density.shape[-2:], orig_shape, sampling) for p in points]

        output = {'points': points, 'probabilities': probabilities, 'sampling': sampling}
        return output
