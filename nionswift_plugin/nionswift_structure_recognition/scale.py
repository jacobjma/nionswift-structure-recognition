import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter

from psm.geometry import regular_polygon, polygon_area
from psm.structures.utils import rotate
from psm.rmsd import pairwise_rmsd
from psm.graph import stable_delaunay_faces


def cosine_window(x, cutoff, rolloff):
    rolloff *= cutoff
    array = .5 * (1 + np.cos(np.pi * (x - cutoff + rolloff) / rolloff))
    array[x > cutoff] = 0.
    array = np.where(x > cutoff - rolloff, array, np.ones_like(x))
    return array


def square_crop(image):
    shape = image.shape

    if image.shape[-1] != min(shape[-2:]):
        n = (image.shape[-2] - image.shape[-1]) // 2
        m = (image.shape[-2] - image.shape[-1]) - n
        image = image[..., n:-m, :]
    elif image.shape[-2] != min(shape[-2:]):
        n = (image.shape[-2] - image.shape[-1]) // 2
        m = (image.shape[-2] - image.shape[-1]) - n
        image = image[..., n:-m]

    return image


def windowed_fft(image):
    #if len(image.shape) == 2:
    #    image = image[None]
    image = square_crop(image)

    x = np.fft.fftshift(np.fft.fftfreq(image.shape[-2]))
    y = np.fft.fftshift(np.fft.fftfreq(image.shape[-1]))
    r = np.sqrt(x[:, None] ** 2 + y[None] ** 2)
    m = cosine_window(r, .5, .33)
    return np.fft.fft2(image * m)


def detect_fourier_spots(image, template, symmetry, min_scale=None, max_scale=None, nbins_angular=128,
                         return_positions=False):
    if symmetry < 2:
        raise RuntimeError('symmetry must be 2 or greater')

    max_max_scale = (min(image.shape[-2:]) // 2) / np.max(np.linalg.norm(template, axis=1))

    if min_scale is None:
        min_scale = 1 / np.min(np.linalg.norm(template, axis=1))

    if max_scale is None:
        max_scale = max_max_scale

    else:
        max_scale = min(max_scale, max_max_scale)

    if min_scale > max_scale:
        raise RuntimeError('min_scale must be less than max_scale')

    f = np.abs(windowed_fft(image))
    if len(f.shape) == 3:
        f = f.mean(0)

    f = np.fft.fftshift(f)

    # import matplotlib.pyplot as plt
    # plt.imshow(f**.1)
    # n = 40
    # plt.xlim(f.shape[0] // 2 - n, f.shape[0] // 2 + n)
    # plt.ylim(f.shape[1] // 2 - n, f.shape[1] // 2 + n)
    # plt.show()
    # sss


    angles = np.linspace(0, 2 * np.pi / symmetry, nbins_angular, endpoint=False)
    scales = np.arange(min_scale, max_scale, 1)

    r = np.linalg.norm(template, axis=1)[:, None, None] * scales[None, :, None]
    a = np.arctan2(template[:, 1], template[:, 0])[:, None, None] + angles[None, None, :]

    templates = np.array([(np.cos(a) * r).ravel(), (np.sin(a) * r).ravel()])
    templates += np.array([f.shape[0] // 2, f.shape[1] // 2])[:, None]

    unrolled = ndimage.map_coordinates(f, templates, order=1)
    unrolled = unrolled.reshape((len(template), len(scales), len(angles)))

    unrolled = (unrolled / unrolled.mean((2,), keepdims=True)).mean(0)
    # unrolled = (unrolled).mean(0)

    # import matplotlib.pyplot as plt
    # plt.imshow(unrolled)
    # plt.show()

    p = np.unravel_index(np.argmax(unrolled), unrolled.shape)

    if return_positions:
        r = np.linalg.norm(template, axis=1) * scales[p[0]]  # + min_scale
        a = np.arctan2(template[:, 1], template[:, 0])
        a -= p[1] * 2 * np.pi / symmetry / nbins_angular + np.pi / symmetry

        spots = np.array([(np.cos(a) * r).ravel(), (np.sin(a) * r).ravel()]).T
        spots += np.array([f.shape[0] // 2, f.shape[1] // 2])[None]
        return scales[p[0]], spots
    else:
        return scales[p[0]]


class FourierSpaceCalibrator:

    def __init__(self, template, lattice_constant, min_sampling=None, max_sampling=None):
        self.template = template
        self.lattice_constant = lattice_constant
        self.min_sampling = min_sampling
        self.max_sampling = max_sampling

    def get_spots(self, image):
        return self.calibrate(image, return_spots=True)[1]

    def calibrate(self, image, return_spots=False):
        if self.template.lower() == 'hexagonal':
            k = min(image.shape[-2:]) / self.lattice_constant * 2 / np.sqrt(3)
            template = regular_polygon(1., 6)
            symmetry = 6
        elif self.template.lower() == '2nd-order-hexagonal':
            k = min(image.shape[-2:]) / self.lattice_constant * 2 / np.sqrt(3)
            template = regular_polygon(1., 6)
            template = np.vstack((template, rotate(template, 30) * np.sqrt(3)))
            symmetry = 6
        else:
            raise NotImplementedError()

        if self.min_sampling is None:
            min_scale = None
        else:
            min_scale = k * self.min_sampling

        if self.max_sampling is None:
            max_scale = None
        else:
            max_scale = k * self.max_sampling

        if return_spots:
            scale, spots = detect_fourier_spots(image, template, symmetry, min_scale=min_scale, max_scale=max_scale,
                                                return_positions=return_spots)
            return scale / k, spots
        else:
            scale = detect_fourier_spots(image, template, symmetry, min_scale=min_scale, max_scale=max_scale,
                                         return_positions=False)
            return scale / k

    def __call__(self, image):
        return self.calibrate(image, return_spots=False)


def detect_scale_real_space(image, model, template, alpha, rmsd_max, min_sampling, max_sampling, step_size=.01):
    max_valid = 0
    best_sampling = None

    for sampling in np.linspace(min_sampling, max_sampling, int(np.ceil((max_sampling - min_sampling) / step_size))):
        points = model(image, sampling)['points']
        if len(points) < 3:
            continue

        faces = stable_delaunay_faces(points, alpha)

        segments = [points[face] for face in faces]
        reference_area = polygon_area(template / sampling)

        rmsd = pairwise_rmsd([template / sampling], segments).ravel()

        # print(sampling)
        # import matplotlib.pyplot as plt
        # plt.plot(*points.T,'o')
        # plt.show()

        valid = rmsd < rmsd_max
        valid = np.where(valid)[0]
        if len(valid) == 0:
            continue

        valid_area = 0.
        for i in valid:
            valid_area += polygon_area(points[faces[i]])

        valid_area_fraction = valid_area / np.prod(image.shape)

        if valid_area_fraction > max_valid:
            area = valid_area / len(valid)
            best_sampling = sampling * np.sqrt(reference_area / area)
            max_valid = valid_area_fraction

    return best_sampling


class RealSpaceCalibrator:

    def __init__(self, model, template, lattice_constant, min_sampling, max_sampling, step_size=.01):
        self.model = model
        self.template = template
        self.lattice_constant = lattice_constant
        self.min_sampling = min_sampling
        self.max_sampling = max_sampling
        self.step_size = step_size

    def __call__(self, image):

        if self.template.lower() == 'hexagonal':
            template = regular_polygon(self.lattice_constant / np.sqrt(3), 6)
            alpha = 2.
            rmsd_max = .05
        else:
            raise NotImplementedError()

        return detect_scale_real_space(image, model=self.model, template=template,
                                       alpha=alpha,
                                       rmsd_max=rmsd_max,
                                       min_sampling=self.min_sampling,
                                       max_sampling=self.max_sampling, step_size=self.step_size)
