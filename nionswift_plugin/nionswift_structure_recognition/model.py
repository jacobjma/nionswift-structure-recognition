import json
import os
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree
from skimage.feature import blob_log
from skimage.measure import label, regionprops, find_contours, approximate_polygon
from skimage.morphology import remove_small_holes
from skimage.draw import polygon2mask
from .filters import gaussian_filter
from .unet import R2UNet, ConvHead
import matplotlib.path as mplPath


def load_preset_model(preset, device=None):
    models_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'models')

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if preset == 'graphene':

        model = AtomRecognitionModel.load(os.path.join(models_dir, 'graphene.json'), device=device)
        return model

    else:
        raise RuntimeError()


def prepare_images(images, device):
    images = torch.tensor(images, device=device)

    if len(images.shape) == 2:
        images = images[None, None]

    if len(images.shape) == 3:
        images = images[:, None]

    images = images.to(device)

    return images


def weighted_normalize(images, weights):
    weighted_images = images * weights
    return (images - weighted_images.mean(axis=(1, 2, 3), keepdims=True)) / \
           weighted_images.std(axis=(1, 2, 3), keepdims=True)


def normalize(images):
    return (images - images.mean(axis=(1, 2, 3), keepdims=True)) / images.std(axis=(1, 2, 3), keepdims=True)


def closest_multiple_ceil(n, m):
    return int(np.ceil(n / m) * m)


def calculate_padding(shape, target_shape, n):
    if n is not None:
        height = closest_multiple_ceil(target_shape[0], n)
        width = closest_multiple_ceil(target_shape[1], n)

    left = max((width - shape[0]) // 2, 0)
    right = max(width - shape[0] - left, 0)
    up = max((height - shape[1]) // 2, 0)
    down = max(height - shape[1] - up, 0)

    padding = [left, right, up, down]
    return padding


def pad_to_size(tensor, target_shape, n=16):
    shape = tensor.shape[-2:]

    padding = calculate_padding(shape, target_shape=target_shape, n=n)
    [left, right, up, down] = padding
    return F.pad(tensor, [up, down, left, right], mode='constant'), padding


def unpad(tensor, padding):
    return tensor[..., padding[0]:tensor.shape[-2] - padding[1], padding[2]:tensor.shape[-1] - padding[3]]


def rescale_centers(centers, old_shape, new_shape, padding):
    scale_factor = (old_shape[0] / (new_shape[0] - padding[0] - padding[1]),
                    old_shape[1] / (new_shape[1] - padding[2] - padding[3]))

    centers = (centers - (padding[0], padding[2])) * scale_factor
    return centers


def is_position_inside_image(positions, shape, margin=0.):
    mask = ((positions[:, 0] >= -margin) & (positions[:, 1] >= -margin) &
            (positions[:, 0] < shape[0] + margin) & (positions[:, 1] < shape[1] + margin))
    return mask


def mask_image(image, contours):
    masked_image = image.copy()

    mask = np.zeros(masked_image.shape, dtype=np.bool)
    for contour in contours:
        mask += polygon2mask(masked_image.shape, contour[:, ::-1])

    masked_image[mask] = image[mask == 0].mean()

    return masked_image


def box_is_inside(box1, box2):
    if box1[0] <= box2[0]:
        return True

    if box1[1] <= box2[1]:
        return True

    if box1[2] >= box2[2]:
        return True

    if box1[3] >= box2[3]:
        return True

    return False


class AtomRecognitionModel:

    def __init__(self, net, train_sampling, mean_bondlength, prefilter_sigma, density_sigma, contaminant_sigma, margin):
        self._net = net
        self._train_sampling = train_sampling
        self._mean_bondlength = mean_bondlength
        self._prefilter_sigma = prefilter_sigma
        self._density_sigma = density_sigma
        self._contaminant_sigma = contaminant_sigma
        self._margin = margin

    @property
    def train_sampling(self):
        return self._train_sampling

    @property
    def device(self):
        return next(self._net.parameters()).device

    @property
    def is_cuda(self):
        return next(self._net.parameters()).is_cuda

    @classmethod
    def load(cls, path, device):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        with open(path, 'r') as fp:
            state = json.load(fp)

        folder = os.path.dirname(path)

        net = ConvHead(R2UNet(state['net']['in_channels'], state['net']['features']), state['net']['out_channels'])

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        net.to(device)
        net.eval()
        net.load_state_dict(torch.load(os.path.join(folder, state['net']['weights_file']), map_location=device))

        return cls(net,
                   train_sampling=state['train_sampling'],
                   mean_bondlength=state['mean_bondlength'],
                   prefilter_sigma=state['prefilter_sigma'],
                   density_sigma=state['density_sigma'],
                   contaminant_sigma=state['contaminant_sigma'],
                   margin=state['margin']
                   )

    def get_max_fov(self):
        available_memory = torch.cuda.get_device_properties(0).total_memory
        max_memory = .9 * available_memory
        max_elements = (1024 * 1024) * max_memory / 1448010752
        return max_elements * self.train_sampling ** 2

    def _resize_images(self, images, sampling):
        scale_factor = sampling / self.train_sampling
        images = F.interpolate(images, scale_factor=scale_factor, mode='area', recompute_scale_factor=True)
        return images

    def _predict_densities(self, images, sampling):
        with torch.no_grad():
            images = prepare_images(images, self.device)

            images = gaussian_filter(images, self._prefilter_sigma / sampling)

            images = self._resize_images(images, sampling)

            images = normalize(images)

            target_shape = (images.shape[2] + self._margin / self.train_sampling,
                            images.shape[3] + self._margin / self.train_sampling)

            padded_images, padding = pad_to_size(images, target_shape)

            densities = torch.sigmoid(self._net(padded_images))

            densities = unpad(densities, padding)

            images = weighted_normalize(images, 1 - densities[:, 1][None])

            padded_images, padding = pad_to_size(images, target_shape)

            densities = torch.sigmoid(self._net(padded_images))

        return densities.detach().cpu().numpy(), padding

    def _predict_lattice(self, density):
        return blob_log(density, 2.5, 3.5)[:, :2]

    def _predict_contaminants(self, density):
        intensity_per_contaminant = (self._contaminant_sigma * np.sqrt(2 * np.pi)) ** 2
        dopant_area = np.pi * self._contaminant_sigma ** 2

        extended_threshold = 5 * intensity_per_contaminant

        segments = density > .1
        segments = remove_small_holes(segments, dopant_area)

        labels = label(segments)
        props = regionprops(labels, density)

        margin = int(self._margin / self.train_sampling)
        image_box = (margin, margin, density.shape[-2] - margin, density.shape[-1] - margin)

        extended_regions = []
        single_contaminants = []

        for prop in props:

            if prop.weighted_moments[0, 0] > extended_threshold:
                extended_regions.append(prop)
                continue

            if prop.weighted_moments[0, 0] < (.5 * intensity_per_contaminant):
                continue

            if box_is_inside(prop.bbox, image_box):
                extended_regions.append(prop)
                continue

            single_contaminants.append(prop)

        contours = []
        for extended_region in extended_regions:
            contour = find_contours(np.pad(extended_region.image, ((1, 1), (1, 1))), .5)[0] + extended_region.bbox[:2]
            contour = approximate_polygon(contour, tolerance=self._mean_bondlength)

            contours.append(contour)

        positions = []
        for contaminant in single_contaminants:
            new_positions = blob_log(contaminant.intensity_image,
                                     min_sigma=self._contaminant_sigma - 1,
                                     max_sigma=self._contaminant_sigma + 1,
                                     overlap=1)[:, :2]
            new_positions += contaminant.bbox[:2]
            positions.append(new_positions)

        if len(positions) > 0:
            positions = np.vstack(positions)
        else:
            positions = np.zeros((0, 2), dtype=np.float32)

        return contours, positions

    def _merge_positions(self, lattice_positions, contaminant_positions):
        if len(contaminant_positions) == 0:
            return lattice_positions, np.zeros(len(lattice_positions), dtype=np.int)

        if len(lattice_positions) == 0:
            return contaminant_positions, np.ones(len(lattice_positions), dtype=np.int)

        distances, overlapping_atoms = KDTree(lattice_positions).query(contaminant_positions, 1)
        # overlapping_atoms = overlapping_atoms[distances < self._mean_bondlength / self.train_sampling * .5]

        to_remove = []
        for i, (distance, overlapping_atom) in enumerate(zip(distances, overlapping_atoms)):
            if distance < (self._mean_bondlength / self.train_sampling * .5):
                contaminant_positions[i] = lattice_positions[overlapping_atom]
                to_remove.append(overlapping_atom)

        lattice_positions = np.delete(lattice_positions, overlapping_atoms, axis=0)
        positions = np.vstack([lattice_positions, contaminant_positions])

        labels = np.zeros(len(positions), dtype=np.int)
        labels[-len(contaminant_positions):] = 1
        return positions, labels

    def predict(self, images, sampling):
        densities, padding = self._predict_densities(images, sampling)

        point_series = []
        labels_series = []
        contours_series = []
        for i in range(len(densities)):
            lattice_positions = self._predict_lattice(densities[i, 0])
            contours, contaminant_positions = self._predict_contaminants(densities[i, 1])

            positions, labels = self._merge_positions(lattice_positions, contaminant_positions)

            valid = np.ones(len(positions), dtype=np.bool)
            for contour in contours:
                valid[[mplPath.Path(contour).contains_point(position) for position in positions]] = False

            positions = rescale_centers(positions, images.shape[-2:], densities.shape[-2:], padding)[:, ::-1]

            for i, contour in enumerate(contours):
                contours[i] = rescale_centers(contour, images.shape[-2:], densities.shape[-2:], padding)[:, ::-1]

            is_inside = is_position_inside_image(positions, images.shape[-2:],
                                                 (self._margin - self._mean_bondlength / 2) / self.train_sampling)

            valid = valid * is_inside

            positions = positions[valid]
            labels = labels[valid]

            positions = positions

            point_series.append(positions)
            labels_series.append(labels)
            contours_series.append(contours)

        densities = unpad(densities, padding)

        result = {'points': point_series,
                  'labels': labels_series,
                  'contours': contours_series,
                  'lattice_densities': densities[:, 0],
                  'contaminant_densities': densities[:, 1]}

        if len(densities) == 1:
            result = {key: value[0] for key, value in result.items()}

        return result

    def __call__(self, image, sampling):
        if len(image.shape) != 2:
            raise RuntimeError()

        if self.is_cuda:
            fov = np.prod(image.shape) * sampling ** 2
            max_fov = self.get_max_fov()
            if fov > max_fov:
                print('predicted fov area {:.3f} Angstrom^2 exceeds maximum ({:.3f} Å^2)'.format(fov, max_fov))
                return None

        try:
            return self.predict(image, sampling=sampling)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('WARNING: ran out of memory')
                torch.cuda.empty_cache()

                return None
            else:
                raise e
