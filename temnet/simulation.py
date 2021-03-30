import numpy as np
from abtem.potentials import superpose_deltas
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from ase import Atoms
from numbers import Number

gaussian = lambda x, sigma: np.exp(-x ** 2 / (2 * sigma ** 2))
lorentzian = lambda x, gamma: gamma / 2 / (np.pi * (x ** 2 + (gamma / 2) ** 2)) / (2 / (np.pi * gamma))


def probe_profile(gaussian_width, lorentzian_width, intensity_ratio=.5, n=100):
    gaussian_width = gaussian_width / 2.355

    x = np.linspace(0, 4 * lorentzian_width, n)
    profile = gaussian(x, gaussian_width) + intensity_ratio * lorentzian(x, lorentzian_width)

    profile /= profile.max()
    f = interp1d(x, profile, fill_value=0, bounds_error=False)
    return f


def bandpass_noise(inner_cutoff, outer_cutoff, shape):
    if isinstance(inner_cutoff, Number):
        inner_cutoff = 1

    kx = np.fft.fftfreq(shape[0]) / cutoff[0] / .5
    ky = np.fft.fftfreq(shape[1]) / cutoff[1] / .5
    k = np.sqrt(kx[None] ** 2 + ky[:, None] ** 2)
    r = np.random.rand(*k.shape).astype(np.float32)
    mask = ((k > inner) & (k < outer)).astype(np.float32)
    noise = np.fft.fftn(mask * np.exp(-1.j * r * 2 * np.pi), axes=tuple(range(len(k.shape))))
    noise = (noise.real + noise.imag) / 2
    return (noise / (np.std(noise) + 1e-6))


def independent_roll(array, shifts):
    shifts[shifts < 0] += array.shape[1]
    x = np.arange(array.shape[0])[:, None]
    y = np.arange(array.shape[1])[None] - shifts[:, None]
    result = array[x, y]
    return result


def make_amourphous_contamination(extent, filling, scale, density, margin=0, sampling=.05):
    shape = np.ceil(np.array((extent[0] + 2 * margin, extent[1] + 2 * margin)) / sampling).astype(np.int)
    sigma = np.max(np.array(shape) * scale) / 2

    noise = gaussian_filter(np.random.randn(*shape), sigma)

    hist, bin_edges = np.histogram(noise, bins=128)
    threshold = bin_edges[np.searchsorted(np.cumsum(hist), noise.size * (1 - filling))]

    contamination = np.zeros_like(noise)
    contamination[noise > threshold] += noise[noise > threshold] - threshold
    contamination = contamination / np.sum(contamination) * density * np.sum(contamination > 0.) * sampling ** 2

    positions = np.array(np.where(np.random.poisson(contamination) > 0.)).astype(np.float).T
    positions += np.random.randn(len(positions), 2)

    positions *= sampling
    positions -= margin
    positions = np.hstack((positions, np.zeros((len(positions), 1))))

    atoms = Atoms('C' * len(positions), positions=positions, cell=[extent[0], extent[1], 0])
    return atoms


def simulate_2d_material(atoms, shape, probe_profile, power_law, eccentricity=0., angle=0.):
    """
    Simulate a STEM image of a 2d material using the convolution approximation.

    Parameters
    ----------
    atoms : ASE Atoms object
        The 2d structure to simulate.
    shape : two ints
        The shape of the output image.
    probe_profile : Callable
        Function for calculating the probe profile.
    power_law : float
        The assumed Z-contrast powerlaw

    Returns
    -------
    ndarray
        Simulated STEM image.
    """

    extent = np.diag(atoms.cell)[:2]
    sampling = extent / shape

    margin = int(np.ceil(5 / min(sampling)))
    shape_w_margin = (shape[0] + 2 * margin, shape[1] + 2 * margin)

    x = np.fft.fftfreq(shape_w_margin[0]) * shape_w_margin[1] * sampling[0]
    y = np.fft.fftfreq(shape_w_margin[1]) * shape_w_margin[1] * sampling[1]

    if eccentricity > 0.:
        r = np.sqrt((x[:, None] / (1 + .5 * eccentricity * (1 + np.cos(angle)))) ** 2 +
                    (y[None] / (1 + .5 * eccentricity * (1 - np.sin(angle)))) ** 2)
    else:
        r = np.sqrt(x[:, None] ** 2 + y[None] ** 2)

    intensity = probe_profile(r)

    positions = atoms.positions[:, :2] / sampling  # - .5

    inside = ((positions[:, 0] > -margin) &
              (positions[:, 1] > -margin) &
              (positions[:, 0] < shape[0] + margin) &
              (positions[:, 1] < shape[1] + margin))

    positions = positions[inside] + margin
    numbers = atoms.numbers[inside]

    array = np.zeros((1,) + shape_w_margin)
    for number in np.unique(atoms.numbers):
        temp = np.zeros((1,) + shape_w_margin)
        superpose_deltas(positions[numbers == number], 0, temp)
        array += temp * number ** power_law

    array = np.fft.ifft2(np.fft.fft2(array) * np.fft.fft2(intensity)).real
    array = array[0, margin:-margin, margin:-margin]
    array = np.abs(array)
    return array


def superpose_gaussians(positions, shape, sigma):
    margin = int(np.ceil(4 * sigma))

    shape_w_margin = (shape[0] + 2 * margin, shape[1] + 2 * margin)
    array = np.zeros((1,) + shape_w_margin)

    x = np.fft.fftfreq(shape_w_margin[0]) * shape_w_margin[1]
    y = np.fft.fftfreq(shape_w_margin[1]) * shape_w_margin[1]

    r = np.sqrt(x[:, None] ** 2 + y[None] ** 2)
    intensity = gaussian(r, sigma)

    inside = ((positions[:, 0] > -margin) &
              (positions[:, 1] > -margin) &
              (positions[:, 0] < shape[0] + margin) &
              (positions[:, 1] < shape[1] + margin))

    positions += margin

    positions = positions[inside]

    superpose_deltas(positions, 0, array)
    array = np.fft.ifft2(np.fft.fft2(array) * np.fft.fft2(intensity)).real
    array = array[0, margin:-margin, margin:-margin]
    return array
