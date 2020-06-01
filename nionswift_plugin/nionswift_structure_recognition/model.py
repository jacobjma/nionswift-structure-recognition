import json
import os
import pathlib

import numpy as np
import skimage.measure
import skimage.util
import skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from .unet import R2UNet, ConvHead


def label_to_index_generator(labels, first_label=0):
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(first_label, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield np.sort(indices[l:h])


def closest_multiple_ceil(n, m):
    return int(np.ceil(n / m) * m)


def pad_to_size(images, height, width, n=16):
    shape = images.shape[-2:]

    if n is not None:
        height = closest_multiple_ceil(height, n)
        width = closest_multiple_ceil(width, n)

    up = max((height - shape[0]) // 2, 0)
    down = max(height - shape[0] - up, 0)
    left = max((width - shape[1]) // 2, 0)
    right = max(width - shape[1] - left, 0)

    padding = [left, right, up, down]

    return F.pad(images, padding, mode='reflect'), padding


def weighted_normalization(image, mask=None):
    if mask is None:
        return (image - torch.mean(image)) / torch.std(image)

    weighted_means = (torch.sum(image * mask, dim=(1, 2, 3), keepdims=True) /
                      torch.sum(mask, dim=(1, 2, 3), keepdims=True))
    weighted_stds = torch.sqrt(
        torch.sum(mask * (image - weighted_means) ** 2, dim=(1, 2, 3), keepdims=True) /
        torch.sum(mask ** 2, dim=(1, 2, 3), keepdims=True))
    return (image - weighted_means) / weighted_stds


def mask_outside_points(points, shape, margin=0):
    mask = ((points[:, 0] >= margin) & (points[:, 1] >= margin) &
            (points[:, 0] < shape[0] - margin) & (points[:, 1] < shape[1] - margin))
    return mask


def merge_close_points(points, distance):
    if len(points) < 2:
        return points, np.arange(len(points))

    clusters = fcluster(linkage(pdist(points), method='complete'), distance, criterion='distance')
    new_points = np.zeros_like(points)
    indices = np.zeros(len(points), dtype=np.int)
    k = 0
    for i, cluster in enumerate(label_to_index_generator(clusters, 1)):
        new_points[i] = np.mean(points[cluster], axis=0)
        indices[i] = np.min(indices)
        k += 1
    return new_points[:k], indices[:k]


def markers_to_points(markers, threshold=.5, merge_distance=0.1):
    points = np.array(np.where(markers > threshold)).T
    if len(points) > 1:
        points, _ = merge_close_points(points, merge_distance)
    return points


def index_array_with_points(points, array, outside_value=0):
    values = np.full(len(points), outside_value, dtype=array.dtype)
    rounded = np.round(points).astype(np.int)
    inside = mask_outside_points(rounded, array.shape)
    inside_points = rounded[inside]
    values[inside] = array[inside_points[:, 0], inside_points[:, 1]]
    return values


def disc_indices(radius):
    X = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.int32)
    x = np.linspace(0, 2 * radius, 2 * radius + 1)
    X[:] = np.linspace(0, 2 * radius, 2 * radius + 1)
    X -= radius

    Y = X.copy().T.ravel()
    X = X.ravel()

    x = x - radius
    r2 = (x[:, None] ** 2 + x[None] ** 2).ravel()
    return X[r2 < radius ** 2], Y[r2 < radius ** 2], r2[r2 < radius ** 2]


def integrate_discs(points, array, radius):
    points = np.round(points).astype(np.int)
    X, Y, r2 = disc_indices(radius)
    weights = np.exp(-r2 / (2 * (radius / 3) ** 2))

    probabilities = np.zeros((len(points), 4))
    for i, point in enumerate(points):
        X_ = point[0] + X
        Y_ = point[1] + Y
        inside = ((X_ > 0) & (X_ < array.shape[1]) & (Y_ > 0) & (Y_ < array.shape[2]))

        X_ = X_[inside]
        Y_ = Y_[inside]
        probabilities[i] = np.sum(array[:, X_, Y_] * weights[None, inside], axis=1)

    return probabilities


def merge_dopants_into_contamination(segmentation):
    binary = segmentation != 0
    labels, n = skimage.measure.label(binary, return_num=True)

    new_segmentation = np.zeros_like(segmentation)
    for label in range(1, n + 1):
        in_segment = labels == label
        if np.sum(segmentation[in_segment] == 1) > np.sum(segmentation[in_segment] == 2):
            new_segmentation[in_segment] = 1
        else:
            new_segmentation[in_segment] = 2

    return new_segmentation


def load_preset_model(preset, device=None):
    models_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'models')

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if preset == 'graphene':

        model = AtomRecognitionModel.load(os.path.join(models_dir, 'graphene.json'), device=device)
        return model

    else:
        raise RuntimeError()


class SeparableFilter(nn.Module):

    def __init__(self, kernel):
        super().__init__()
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        x = F.pad(x, list((len(self.kernel) // 2,) * 4))
        return F.conv2d(F.conv2d(x, self.kernel.reshape((1, 1, 1, -1))), self.kernel.reshape((1, 1, -1, 1)))


class GaussianFilter2d(SeparableFilter):
    def __init__(self, sigma):
        kernel_size = int(np.ceil(sigma)) * 8 + 1
        A = 1 / (sigma * np.sqrt(2 * np.pi))
        kernel = A * torch.exp(-(torch.arange(kernel_size) - (kernel_size - 1) / 2) ** 2 / (2 * sigma ** 2))
        super().__init__(kernel)


class SumFilter2d(SeparableFilter):
    def __init__(self, kernel_size):
        kernel = torch.ones(kernel_size)
        super().__init__(kernel)


class PeakEnhancementFilter(nn.Module):

    def __init__(self, alpha, sigmas, iterations, epsilon=1e-7):
        super().__init__()
        self._filters = nn.ModuleList([GaussianFilter2d(sigma) for sigma in sigmas])
        self._alpha = alpha
        self._iterations = iterations
        self._epsilon = epsilon

    def forward(self, tensor):
        temp = tensor.clone()
        for i in range(self._iterations):
            temp = temp ** self._alpha
            for filt in self._filters:
                temp = temp * filt(tensor) / (filt(temp) + self._epsilon)
        return temp


def threshold_otsu(tensor, nbins=256):
    min_value = tensor.min()
    max_value = tensor.max()

    hist = torch.histc(tensor, bins=nbins, min=min_value, max=max_value)

    bin_centers = torch.linspace(min_value, max_value, nbins).to(tensor)

    weight1 = torch.cumsum(hist, 0)
    weight2 = torch.flip(torch.cumsum(torch.flip(hist, (0,)), 0), (0,))

    mean1 = torch.cumsum(hist * bin_centers, 0) / weight1
    mean2 = torch.flip(torch.cumsum(torch.flip(hist * bin_centers, (0,)), 0) / torch.flip(weight2, (0,)), (0,))

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = torch.argmax(variance12)
    threshold = bin_centers[:-1][idx]

    return threshold.item()


class AtomRecognitionModel:

    def __init__(self, backbone, density_head, segmentation_head, training_sampling, density_sigma, threshold,
                 enhancement_filter_kwargs, device):
        self._backbone = backbone
        self._segmentation_head = segmentation_head
        self._density_head = density_head
        self._training_sampling = training_sampling
        self._density_sigma = density_sigma
        self._threshold = threshold
        self._enhancement_filter = PeakEnhancementFilter(**enhancement_filter_kwargs)
        self._device = device
        if device is not None:
            self.to(device)

    def to(self, *args, **kwargs):
        self._backbone.to(*args, **kwargs)
        self._segmentation_head.to(*args, **kwargs)
        self._density_head.to(*args, **kwargs)
        self._enhancement_filter.to(*args, **kwargs)
        return self

    @classmethod
    def load(cls, path, device=None):

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        with open(path, 'r') as fp:
            state = json.load(fp)

        folder = os.path.dirname(path)

        backbone = R2UNet(1, 6)
        density_head = ConvHead(backbone.out_type, 1)
        segmentation_head = ConvHead(backbone.out_type, state['segmentation_head']['num_classes'])

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        backbone.load_state_dict(
            torch.load(os.path.join(folder, state['backbone']['weights_file']), map_location=device))
        density_head.load_state_dict(
            torch.load(os.path.join(folder, state['density_head']['weights_file']), map_location=device))
        segmentation_head.load_state_dict(
            torch.load(os.path.join(folder, state['segmentation_head']['weights_file']), map_location=device))

        return cls(backbone=backbone,
                   density_head=density_head,
                   segmentation_head=segmentation_head,
                   training_sampling=state['training_sampling'],
                   density_sigma=state['density_sigma'],
                   threshold=state['threshold'],
                   enhancement_filter_kwargs=state['enhancement_filter'],
                   device=device)

    @property
    def training_sampling(self):
        return self._training_sampling

    @property
    def is_cuda(self):
        return next(self._backbone.parameters()).is_cuda

    def prepare_image(self, image, sampling, mask=None):
        image = image.astype(np.float32)

        image = torch.tensor(image)[None, None].to(self._device)

        # image = GaussianFilter2d(2).to(image)(image)
        # print(0.2 / sampling, 1/(sampling / self._training_sampling))

        image = F.interpolate(image, scale_factor=sampling / self._training_sampling, mode='area')

        image, padding = pad_to_size(image, image.shape[2] + 2, image.shape[3] + 2, n=16)

        # if mask is None:
        #    mask = (image < threshold_otsu(image)).type(torch.float32)

        image = weighted_normalization(image, mask)

        return image, sampling, padding

    def get_max_fov(self):
        available_memory = torch.cuda.get_device_properties(0).total_memory
        max_memory = .9 * available_memory
        max_elements = (1024 * 1024) * max_memory / 1448010752
        return max_elements * self.training_sampling ** 2

    def __call__(self, image, sampling, recurrent_normalization=False):
        if len(image.shape) != 2:
            raise RuntimeError()

        if self._device.type.split(':')[0] == 'cuda':
            fov = np.prod(image.shape) * sampling ** 2
            max_fov = self.get_max_fov()
            if fov > max_fov:
                print('predicted fov area {:.3f} Angstrom^2 exceeds maximum ({:.3f} Å^2)'.format(fov, max_fov))
                return None

        try:
            return self.predict(image, sampling=sampling, recurrent_normalization=recurrent_normalization)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('WARNING: ran out of memory')
                torch.cuda.empty_cache()

                return None
            else:
                raise e

    def predict_series(self, images, sampling, stop_event=None):
        if len(images.shape) == 2:
            images = images[None]

        t = .1
        output = []
        for i, image in enumerate(images):
            if stop_event is not None:
                if stop_event.is_set():
                    return None

            output.append(self(image, sampling))

            if i / len(images) > t:
                print('{} of {} frames processed'.format(i + 1, len(images)))
                t += .1

        print('all frames processed')

        return output

    def predict(self, image, sampling, recurrent_normalization=False):
        with torch.no_grad():

            preprocessed, sampling, padding = self.prepare_image(image, sampling)

            if recurrent_normalization:
                segmentation = nn.Softmax(1)(self._segmentation_head(self._backbone(preprocessed)))
                mask = segmentation[:, 1, None] + segmentation[:, 3, None]
                preprocessed, sampling, padding = self.prepare_image(image, sampling, mask)

            features = self._backbone(preprocessed)
            segmentation = nn.Softmax(1)(self._segmentation_head(features))

            density = nn.Sigmoid()(self._density_head(features))

            markers = self._enhancement_filter(density)
            classes = segmentation.argmax(1, keepdim=True)

            contamination = classes == 2
            classes = F.interpolate(classes.type(torch.float32),
                                    scale_factor=self.training_sampling / sampling).type(torch.int)
            density = F.interpolate(density, scale_factor=self.training_sampling / sampling)

            markers = markers[0, 0].detach().cpu().numpy()
            density = density[0, 0].detach().cpu().numpy()
            classes = classes[0, 0].detach().cpu().numpy()
            contamination = contamination[0, 0].detach().cpu().numpy()
            segmentation = segmentation[0].detach().cpu().numpy()

        torch.cuda.empty_cache()

        points = np.array(np.where(markers > self._threshold * self._density_sigma ** 2 * 2 * np.pi)).T
        points, indices = merge_close_points(points, self._density_sigma)

        label_probabilities = integrate_discs(points, segmentation, self._density_sigma)
        labels = np.argmax(label_probabilities, axis=-1)

        points = points.astype(np.float)
        points = points - padding[0]

        if sampling is not None:
            points *= self._training_sampling / sampling

        valid = ((labels != 0) &
                 (labels != 2) &
                 (points[:, 0] > 0) &
                 (points[:, 1] > 0) &
                 (points[:, 0] < image.shape[0]) &
                 (points[:, 1] < image.shape[1]))

        points = points[valid]
        labels = labels[valid]

        if np.any(contamination):
            contamination = skimage.transform.rescale(contamination, .5)
            contamination = np.pad(contamination, ((1, 1), (1, 1)))
            contours = skimage.measure.find_contours(contamination, .9)

            if (len(contours) > 0) & (len(points) > 0):
                contours = (np.vstack(contours) - 1) * 2 - padding[0]
                contours = contours[::10] * self._training_sampling / sampling
                points = np.vstack((points, contours))
                labels = np.concatenate((labels, np.full(len(contours), 2)))

        output = {'points': points[:, ::-1],
                  'labels': labels,
                  'density': density,
                  'segmentation': classes,
                  'sampling': sampling}

        return output
