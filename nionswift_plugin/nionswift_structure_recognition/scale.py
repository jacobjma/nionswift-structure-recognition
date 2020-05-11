import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

from .geometry import regular_polygon, polygon_area, pairwise_rmsd
from .geometry import rotate
from .graph import stable_delaunay_faces


def cosine_window(x, cutoff, rolloff):
    rolloff *= cutoff
    array = .5 * (1 + np.cos(np.pi * (x - cutoff + rolloff) / rolloff))
    array[x > cutoff] = 0.
    array = np.where(x > cutoff - rolloff, array, np.ones_like(x))
    return array


def square_crop(image):
    shape = image.shape

    if image.shape[0] != min(shape):
        n = (image.shape[0] - image.shape[1]) // 2
        m = (image.shape[0] - image.shape[1]) - n
        image = image[n:-m]
    elif image.shape[1] != min(shape):
        n = (image.shape[1] - image.shape[0]) // 2
        m = (image.shape[1] - image.shape[0]) - n
        image = image[:, n:-m]

    return image


def windowed_fft(image):
    image = square_crop(image)
    x = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))
    y = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))
    r = np.sqrt(x[:, None] ** 2 + y[None] ** 2)
    m = cosine_window(r, .5, .33)
    return np.fft.fft2(image * m)


def detect_scale_fourier_space(image, template, symmetry, min_scale=None, max_scale=None, nbins_angular=16):
    if symmetry < 2:
        raise RuntimeError('symmetry must be 2 or greater')

    min_min_scale = 1 / np.min(np.linalg.norm(template, axis=1))
    max_max_scale = (min(image.shape) // 2) / np.max(np.linalg.norm(template, axis=1))

    if min_scale is None:
        min_scale = min_min_scale

    else:
        min_scale = min(min_scale, min_min_scale)

    if max_scale is None:
        max_scale = max_max_scale

    else:
        max_scale = min(max_scale, max_max_scale)

    if min_scale > max_scale:
        raise RuntimeError('min_scale must be less than max_scale')

    f = windowed_fft(image)
    f = np.log(np.abs(np.fft.fftshift(f)))
    f = gaussian_filter(f, 1)

    angles = np.linspace(0, 2 * np.pi / symmetry, nbins_angular, endpoint=False)
    scales = np.arange(min_scale, max_scale, 1)

    r = np.linalg.norm(template, axis=1)[:, None, None] * scales[None, :, None]
    a = np.arctan2(template[:, 1], template[:, 0])[:, None, None] + angles[None, None, :]

    templates = np.array([np.cos(a) * r, np.sin(a) * r])
    templates = templates.reshape(2, -1) + np.array([f.shape[0] // 2, f.shape[1] // 2])[:, None]

    unrolled = ndimage.map_coordinates(f, templates, order=1)
    unrolled = unrolled.reshape((len(template), len(scales), len(angles)))
    unrolled = (unrolled.mean(0) / unrolled.mean((0, 2), keepdims=True))[0]

    return np.unravel_index(np.argmax(unrolled), unrolled.shape)[0] + min_scale


class FourierSpaceCalibrator:

    def __init__(self, crystal_system, lattice_constant, min_sampling=None, max_sampling=None):
        self._crystal_system = crystal_system
        self._lattice_constant = lattice_constant
        self._min_sampling = min_sampling
        self._max_sampling = max_sampling

    def __call__(self, image):
        if self._crystal_system.lower() == 'hexagonal':
            k = min(image.shape) / self._lattice_constant * 2 / np.sqrt(3)
            template = regular_polygon(1., 6)
            template = np.vstack((template, rotate(template, np.pi / 6) * np.sqrt(3)))
            symmetry = 6
        else:
            raise NotImplementedError()

        if self._min_sampling is None:
            min_scale = None
        else:
            min_scale = k * self._min_sampling

        if self._max_sampling is None:
            max_scale = None
        else:
            max_scale = k * self._max_sampling

        return detect_scale_fourier_space(image, template, symmetry, min_scale=min_scale, max_scale=max_scale) / k


def detect_scale_real_space(image, model, template, alpha, rmsd_max, min_sampling, max_sampling, step_size=.005):
    reference_area = polygon_area(template)
    max_valid = 0
    min_rmsd = np.inf
    best_sampling = None

    for sampling in np.linspace(min_sampling, max_sampling, int(np.ceil((max_sampling - min_sampling) / step_size))):
        points = model(image, sampling)['points']

        if len(points) < 3:
            continue

        faces = stable_delaunay_faces(points, alpha)

        segments = [points[face] for face in faces]
        rmsd = pairwise_rmsd([template], segments).ravel()

        valid = rmsd < rmsd_max
        num_valid = np.sum(valid)
        if num_valid == 0:
            continue

        if num_valid >= max_valid:
            if rmsd[valid].mean() < min_rmsd:
                area = np.mean([polygon_area(segment) for i, segment in enumerate(segments) if valid[i]])
                best_sampling = sampling / np.sqrt(reference_area / area)

                max_valid = num_valid
                min_rmsd = rmsd[valid].mean()

    return best_sampling


class RealSpaceCalibrator:

    def __init__(self, model, crystal_system, lattice_constant, min_sampling, max_sampling, step_size=.01):
        self._model = model
        self._crystal_system = crystal_system
        self._lattice_constant = lattice_constant
        self._min_sampling = min_sampling
        self._max_sampling = max_sampling
        self._step_size = step_size

    def __call__(self, image):

        if self._crystal_system.lower() == 'hexagonal':
            template = regular_polygon(self._lattice_constant / self._model.training_sampling, 6)
            alpha = 2.
            rmsd_max = .05
        else:
            raise NotImplementedError()

        return detect_scale_real_space(image, model=self._model, template=template,
                                       alpha=alpha,
                                       rmsd_max=rmsd_max,
                                       min_sampling=self._min_sampling,
                                       max_sampling=self._max_sampling, step_size=self._step_size)
