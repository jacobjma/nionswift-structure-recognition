import numpy as np
from scipy.ndimage import gaussian_filter


def repeat(points, cell, n, m):
    N = len(points)

    n0, n1 = 0, n
    m0, m1 = 0, m
    new_points = np.zeros((n * m * N, 2), dtype=np.float)
    new_points[:N] = points

    k = N
    for i in range(n0, n1):
        for j in range(m0, m1):
            if i + j != 0:
                l = k + N
                new_points[k:l] = points + np.dot(np.array((i, j)), cell)
                k = l

    cell = cell * np.array((n, m))

    return new_points, cell


def wrap(points, cell, center=(0.5, 0.5), eps=1e-7):
    if not hasattr(center, '__len__'):
        center = (center,) * 2

    shift = np.asarray(center) - 0.5 - eps

    fractional = np.linalg.solve(cell.T, np.asarray(points).T).T - shift

    for i in range(2):
        fractional[:, i] %= 1.0
        fractional[:, i] += shift[i]

    points = np.dot(fractional, cell)
    return points


def rotate(points, angle, cell=None, center=None, rotate_cell=False):
    if center is None:
        if cell is None:
            center = np.array([0., 0.])
        else:
            center = cell.sum(axis=1) / 2

    angle = angle / 180. * np.pi
    R = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = np.dot(R, points.T - np.array(center)[:, None]).T + center
    if rotate_cell:
        cell = np.dot(R, cell.T).T

    if cell is None:
        return points
    else:
        return points, cell


def fill_rectangle(points, cell, extent, origin=None, margin=0., eps=1e-17):
    if origin is None:
        origin = np.zeros(2)

    P_inv = np.linalg.inv(cell)

    origin_t = np.dot(origin, P_inv)
    origin_t = origin_t % 1.0

    lower_corner = np.dot(origin_t, cell)
    upper_corner = lower_corner + extent

    corners = np.array([[-margin - eps, -margin - eps],
                        [upper_corner[0].item() + margin + eps, -margin - eps],
                        [upper_corner[0].item() + margin + eps, upper_corner[1].item() + margin + eps],
                        [-margin - eps, upper_corner[1].item() + margin + eps]])
    n0, m0 = 0, 0
    n1, m1 = 0, 0
    for corner in corners:
        new_n, new_m = np.ceil(np.dot(corner, P_inv)).astype(np.int)
        n0 = max(n0, new_n)
        m0 = max(m0, new_m)
        new_n, new_m = np.floor(np.dot(corner, P_inv)).astype(np.int)
        n1 = min(n1, new_n)
        m1 = min(m1, new_m)

    points, _ = repeat(points, cell, (1 + n0 - n1).item(), (1 + m0 - m1).item())

    points = points + cell[0] * n1 + cell[1] * m1

    inside = ((points[:, 0] > lower_corner[0] + eps - margin) &
              (points[:, 1] > lower_corner[1] + eps - margin) &
              (points[:, 0] < upper_corner[0] + margin) &
              (points[:, 1] < upper_corner[1] + margin))
    new_points = points[inside] - lower_corner

    cell = np.array([[extent[0], 0], [0, extent[1]]])

    return new_points, cell


def rectangular_graphene(a=2.46, n=1, m=1):
    basis = np.array([(0, 0), (2 / 3., 1 / 3.)])
    cell = np.array([[a, 0], [-a / 2., a * 3 ** 0.5 / 2.]])
    points = np.dot(basis, cell)
    points, cell = repeat(points, cell, 1, 2)
    cell = np.array([[cell[0, 0], 0], [0, cell[1, 1]]])
    points, cell = repeat(points, cell, n, m)
    return wrap(points, cell), cell


def superpose_deltas(positions, shape):
    array = np.zeros(shape)

    rounded = np.floor(positions).astype(np.int)
    inside = (rounded[:,0] > 0) & (rounded[:,1] > 0) & (rounded[:,0] < shape[0]) & (rounded[:,1] < shape[1])
    rounded = rounded[inside]
    positions = positions[inside]
    rows, cols = rounded[:, 0], rounded[:, 1]

    array[rows, cols] += (1 - (positions[:, 0] - rows)) * (1 - (positions[:, 1] - cols))
    array[(rows + 1) % shape[0], cols] += (positions[:, 0] - rows) * (1 - (positions[:, 1] - cols))
    array[rows, (cols + 1) % shape[1]] += (1 - (positions[:, 0] - rows)) * (positions[:, 1] - cols)
    array[(rows + 1) % shape[0], (cols + 1) % shape[1]] += (rows - positions[:, 0]) * (cols - positions[:, 1])
    return array


def bandpass_noise(inner, outer, shape):
    if len(shape) == 1:
        k = np.fft.fftfreq(shape[0], 1 / shape[0])
    elif len(shape) == 2:
        kx = np.fft.fftfreq(shape[0], 1 / shape[0])
        ky = np.fft.fftfreq(shape[1], 1 / shape[1])
        k = np.sqrt(kx[:, None] ** 2 + ky[None] ** 2)
    else:
        raise RuntimeError()

    r = np.random.rand(*k.shape).astype(np.float32)
    mask = ((k > inner) & (k < outer)).astype(np.float32)
    noise = np.fft.fftn(mask * np.exp(-1.j * r * 2 * np.pi), axes=tuple(range(len(k.shape))))
    noise = (noise.real + noise.imag) / 2
    return (noise / (np.std(noise) + 1e-6))


def simulated_graphene(shape, sampling, sigma, noise_level, rotation=None, contamination=0., background=0., seed=None):
    if seed is not None:
        np.random.seed(seed)

    margin = 2.46
    sigma = sigma / sampling
    points, cell = rectangular_graphene(n=2, m=1)

    if rotation is None:
        rotation = np.random.rand() * 360

    points, cell = rotate(points, rotation, cell, rotate_cell=True)

    points, cell = fill_rectangle(points, cell, (shape[0] * sampling, shape[1] * sampling), margin=margin)

    margin = int(np.ceil(2.46 / sampling))
    points = points / sampling + margin

    if contamination > 0.:
        noise = bandpass_noise(0, 2, (shape[0] + 2 * margin, shape[1] + 2 * margin))

        hist, bin_edges = np.histogram(noise, bins=128)
        threshold = bin_edges[np.searchsorted(np.cumsum(hist), noise.size * (1 - contamination))]

        contamination = np.zeros_like(noise)
        contamination[noise > threshold] += noise[noise > threshold] - threshold
        contamination = contamination ** .5

        contamination_area = np.sum(contamination > 0.) * sampling ** 2
        contamination = contamination / np.sum(contamination) * 5 * contamination_area

        contamination_points = np.array(np.where(np.random.poisson(contamination) > 0.)).astype(np.float).T
        contamination_points += np.random.randn(len(contamination_points), 2)

        rounded = points.astype(np.int)
        points = points[contamination[rounded[:, 0], rounded[:, 1]] == 0.]
        points = np.concatenate((points, contamination_points))

    image = superpose_deltas(points, (shape[0] + 2 * margin, shape[1] + 2 * margin))


    image = gaussian_filter(image, sigma)
    image = image[margin:-margin, margin:-margin]
    image *= (sigma * np.sqrt(2 * np.pi)) ** 2
    image += background
    image /= noise_level
    # image = image / np.sum(image) * dose * sampling ** 2 * np.prod(shape)
    image = np.random.poisson(image).astype(np.float)
    return image
