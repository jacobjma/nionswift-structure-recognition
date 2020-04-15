import json
import os
import pathlib

import numpy as np
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from .utils import StructureRecognitionModule
from .widgets import Section, line_edit_template, combo_box_template


def load_presets():
    presets_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'presets')

    presets = {}
    for file in os.listdir(presets_dir):
        with open(os.path.join(presets_dir, file)) as f:
            new_preset = json.load(f)
            presets[new_preset['name']] = new_preset
    return presets


presets = load_presets()


def periodic_smooth_decomposition(image):
    u = image
    v = u2v(u)
    v_fft = np.fft.fft2(v)
    s = v2s(v_fft)
    s_i = np.fft.ifft2(s)
    s_f = np.real(s_i)
    p = u - s_f  # u = p + s
    return p, s_f


def u2v(u):
    v = np.zeros(u.shape, dtype=u.dtype)
    v[..., 0, :] = u[..., -1, :] - u[..., 0, :]
    v[..., -1, :] = u[..., 0, :] - u[..., -1, :]

    v[..., :, 0] += u[..., :, -1] - u[..., :, 0]
    v[..., :, -1] += u[..., :, 0] - u[..., :, -1]
    return v


def v2s(v_hat):
    M, N = v_hat.shape[-2:]

    q = np.arange(M).reshape(M, 1).astype(v_hat.dtype)
    r = np.arange(N).reshape(1, N).astype(v_hat.dtype)

    den = (2 * np.cos(np.divide((2 * np.pi * q), M)) + 2 * np.cos(np.divide((2 * np.pi * r), N)) - 4)

    for i in range(len(v_hat.shape) - 2):
        den = np.enpand_dims(den, 0)

    s = v_hat / (den + 1e-12)
    s[..., 0, 0] = 0
    return s


def periodic_smooth_decomposed_fft(image):
    p, s = periodic_smooth_decomposition(image)
    return np.fft.fft2(p)


def image_as_polar_representation(image, inner=1, outer=None, symmetry=1, bins_per_symmetry=32):
    center = np.array(image.shape[-2:]) // 2

    if outer is None:
        outer = (np.min(center) // 2).item()

    n_radial = outer - inner
    n_angular = (symmetry // 2) * bins_per_symmetry

    radials = np.linspace(inner, outer, n_radial)
    angles = np.linspace(0, np.pi, n_angular)

    polar_coordinates = center[:, None, None] + radials[None, :, None] * np.array([np.cos(angles), np.sin(angles)])[:,
                                                                         None]
    polar_coordinates = polar_coordinates.reshape((2, -1))
    unrolled = ndimage.map_coordinates(image, polar_coordinates, order=1)
    unrolled = unrolled.reshape((n_radial, n_angular))

    if symmetry > 1:
        unrolled = unrolled.reshape((unrolled.shape[0], symmetry // 2, -1)).sum(1)

    return unrolled


def _window_sum_2d(image, window_shape):
    window_sum = np.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1] - window_sum[:-window_shape[0] - 1])

    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = (window_sum[:, window_shape[1]:-1] - window_sum[:, :-window_shape[1] - 1])

    return window_sum


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def normalized_cross_correlation_with_2d_gaussian(image, kernel_size, std):
    kernel_1d = np.exp(-(np.arange(kernel_size) - (kernel_size - 1) / 2) ** 2 / (2 * std ** 2))
    kernel = np.outer(kernel_1d, kernel_1d)
    kernel_mean = kernel.mean()
    kernel_ssd = np.sum((kernel - kernel_mean) ** 2)

    xcorr = ndimage.convolve(ndimage.convolve(image, kernel_1d[None]), kernel_1d[None].T)

    image_window_sum = _window_sum_2d(image, kernel.shape)
    image_window_sum2 = _window_sum_2d(image ** 2, kernel.shape)

    xcorr = _centered(xcorr, image_window_sum.shape)
    numerator = xcorr - image_window_sum * kernel_mean

    denominator = image_window_sum2
    np.multiply(image_window_sum, image_window_sum, out=image_window_sum)
    np.divide(image_window_sum, np.prod(kernel.shape), out=image_window_sum)
    denominator -= image_window_sum
    denominator *= kernel_ssd
    np.maximum(denominator, 0, out=denominator)
    np.sqrt(denominator, out=denominator)

    response = np.zeros_like(xcorr, dtype=np.float32)
    mask = denominator >= np.finfo(np.float32).eps
    response[mask] = numerator[mask] / (denominator[mask])
    return response  # np.float32(mask)


def find_hexagonal_spots(image, lattice_constant=None, min_sampling=None, max_sampling=None, bins_per_symmetry=32,
                         return_cartesian=False):
    if image.shape[0] != image.shape[1]:
        raise RuntimeError('image is not square')

    n = image.shape[0]

    if (lattice_constant is None) & ((min_sampling is not None) | (max_sampling is not None)):
        raise RuntimeError()

    k = n / lattice_constant * 2 / np.sqrt(3)
    if min_sampling is None:
        inner = 1
    else:
        inner = int(np.ceil(max(1, k * min_sampling)))

    if max_sampling is None:
        outer = None
    else:
        outer = int(np.floor(min(n // 2, k * max_sampling)))

    f = periodic_smooth_decomposed_fft(image)
    f[0, 0] = 0
    f = np.abs(np.fft.fftshift(f)) ** 2
    f = (f - f.min()) / (f.max() - f.min())

    peaks = []
    for w in range(5, 11, 1):
        response = normalized_cross_correlation_with_2d_gaussian(f, w, w / 8)
        polar = image_as_polar_representation(response, inner=inner, outer=outer, symmetry=6,
                                              bins_per_symmetry=bins_per_symmetry)
        peak = np.vstack(np.unravel_index((-polar).ravel().argsort()[:5], polar.shape)).T
        peaks.append(peak)

    peaks = np.asarray(np.vstack(peaks))

    below_center = peaks[:, 1] > bins_per_symmetry / 2

    peaks = np.vstack([peaks,
                       np.array([peaks[below_center][:, 0], peaks[below_center][:, 1] - bins_per_symmetry]).T,
                       np.array([peaks[below_center == 0][:, 0],
                                 peaks[below_center == 0][:, 1] + bins_per_symmetry]).T])

    assignments = fcluster(linkage(pdist(peaks), method='single'), 2, 'distance')
    unique, counts = np.unique(assignments, return_counts=True)
    cluster_centers = np.array([np.mean(peaks[assignments == u], axis=0) for u in unique])

    valid = (cluster_centers[:, 1] > -1e-12) & (cluster_centers[:, 1] < bins_per_symmetry) & (counts > 3)
    cluster_centers = cluster_centers[valid][np.argsort(-counts[valid])][:2]

    cluster_centers[:, 0] += inner - .5
    cluster_centers[:, 1] = (cluster_centers[:, 1] / (bins_per_symmetry * 6) * 2 * np.pi) % (2 * np.pi / 6)

    if len(cluster_centers) > 1:
        radial_ratio = np.min(cluster_centers[:, 0]) / np.max(cluster_centers[:, 0])
        angle_diff = np.max(cluster_centers[:, 1]) - np.min(cluster_centers[:, 1])
        if (np.abs(radial_ratio * np.sqrt(3) - 1) < .1) & (np.abs(angle_diff - np.pi / 6) < np.pi / 10):
            spot_radial, spot_angle = cluster_centers[np.argmin(cluster_centers[:, 0])]
        else:
            spot_radial, spot_angle = cluster_centers[0]

    else:
        spot_radial, spot_angle = cluster_centers[0]

    if return_cartesian:
        angles = spot_angle + np.linspace(0, 2 * np.pi, 6, endpoint=False)
        radial = np.array(spot_radial)[None]
        return spot_radial, spot_angle, np.array([np.cos(angles) * radial + image.shape[0] // 2,
                                                  np.sin(angles) * radial + image.shape[0] // 2]).T
    else:
        return spot_radial, spot_angle


def find_hexagonal_sampling(image, lattice_constant, min_sampling=None, max_sampling=None):

    if len(image.shape) > 2:
        raise RuntimeError()

    if image.shape[0] != image.shape[1]:
        raise RuntimeError('image is not square')

    n = image.shape[0]

    radial, _ = find_hexagonal_spots(image, lattice_constant, min_sampling, max_sampling)
    return (radial * lattice_constant / n * np.sqrt(3) / 2).item()


class ScaleDetectionModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

        self.crystal_system = None
        self.lattice_constant = None

    def create_widgets(self, column):
        section = Section(self.ui, 'Scale detection')
        column.add(section)

        lattice_constant_row, self.lattice_constant_line_edit = line_edit_template(self.ui, 'Lattice constant [Å]')
        min_sampling_row, self.min_sampling_line_edit = line_edit_template(self.ui, 'Min. sampling [Å / pixel]')
        crystal_system_row, self.crystal_system_combo_box = combo_box_template(self.ui, 'Crystal system', ['Hexagonal'])

        section.column.add(crystal_system_row)
        section.column.add(lattice_constant_row)
        section.column.add(min_sampling_row)

    def set_preset(self, name):
        self.crystal_system_combo_box.current_item = presets[name]['scale']['crystal_system']
        self.lattice_constant_line_edit.text = presets[name]['scale']['lattice_constant']
        self.min_sampling_line_edit.text = presets[name]['scale']['min_sampling']

    def fetch_parameters(self):
        self.crystal_system = self.crystal_system_combo_box._widget.current_item.lower()
        self.lattice_constant = float(self.lattice_constant_line_edit._widget.text)
        self.min_sampling = float(self.min_sampling_line_edit._widget.text)

    def detect_scale(self, data):
        if self.crystal_system not in ['hexagonal']:
            raise RuntimeError('structure {} not recognized for scale recognition'.format(self.crystal_system))

        scale = find_hexagonal_sampling(data, lattice_constant=self.lattice_constant, min_sampling=self.min_sampling)
        return scale
