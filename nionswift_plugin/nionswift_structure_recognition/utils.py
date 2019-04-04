import numpy as np
import scipy

import tensorflow as tf
from pomegranate import MultivariateGaussianDistribution, GeneralMixtureModel


def rescale(images, scale):
    new_shape = np.round(np.array(images.shape[1:-1]) * scale).astype(int)
    return tf.image.resize(images, new_shape)



def pad_to_closest_multiple(images, m):
    target_height = int(np.ceil(images.shape[1] / m) * m)
    target_width = int(np.ceil(images.shape[2] / m) * m)
    images = tf.image.pad_to_bounding_box(images, 0, 0, target_height, target_width)
    return images


def normalize_global(images):
    moments = tf.nn.moments(images, axes=[1, 2], keepdims=True)
    return (images - moments[0]) / tf.sqrt(moments[1])


def gaussian_kernel(sigma, truncate=None, dtype=tf.float32):
    if truncate is None:
        truncate = tf.math.ceil(3 * tf.cast(sigma, dtype))
    x = tf.cast(tf.range(-truncate, truncate + 1), dtype)
    return 1. / (np.sqrt(2. * np.pi) * sigma) * tf.exp(-x ** 2 / (2 * sigma ** 2))


def gaussian_filter(image, sigma, truncate=None):
    kernel = gaussian_kernel(sigma, truncate)
    image = tf.nn.conv2d(image, kernel[..., None, None, None], strides=[1, 1, 1, 1], padding='SAME')
    image = tf.nn.conv2d(image, kernel[None, ..., None, None], strides=[1, 1, 1, 1], padding='SAME')
    return image


def normalize_local(images, sigma):
    truncate = tf.cast(tf.math.ceil(3 * tf.cast(sigma, tf.float32)), tf.int32)
    images = tf.pad(images, [[0, 0], [truncate, truncate], [truncate, truncate], [0, 0]], 'REFLECT')
    mean = gaussian_filter(images, sigma, truncate)
    images = images - mean
    images = images / tf.sqrt(gaussian_filter(images ** 2, sigma, truncate))
    return images[:, truncate:-truncate, truncate:-truncate, :]


def ensemble_expand(image):
    ensemble = np.zeros((8,) + image.shape)

    ensemble[0] = image
    ensemble[1] = np.fliplr(image)
    ensemble[2] = np.flipud(image)
    ensemble[3] = np.rot90(image)
    ensemble[4] = np.fliplr(np.flipud(image))
    ensemble[5] = np.fliplr(np.rot90(image))
    ensemble[6] = np.fliplr(np.flipud(np.rot90(image)))
    ensemble[7] = np.flipud(np.rot90(image))

    return ensemble


def ensemble_reduce(ensemble):
    ensemble[1] = np.fliplr(ensemble[1])
    ensemble[2] = np.flipud(ensemble[2])
    ensemble[3] = np.rot90(ensemble[3], k=3)
    ensemble[4] = np.flipud(np.fliplr(ensemble[4]))
    ensemble[5] = np.rot90(np.fliplr(ensemble[5]), k=3)
    ensemble[6] = np.rot90(np.flipud(np.fliplr(ensemble[6])), k=3)
    ensemble[7] = np.rot90(np.flipud(ensemble[7]), k=3)

    return np.sum(ensemble, axis=0) / 8.


def labeled_comprehension(input, labels, index, func, output_shape):
    as_scalar = np.isscalar(index)
    input = np.asarray(input)

    positions = np.arange(input.size).reshape(input.shape)

    if labels is None:
        if index is not None:
            raise ValueError("index without defined labels")

        return func(input.ravel(), positions.ravel())

    try:
        input, labels = np.broadcast_arrays(input, labels)
    except ValueError:
        raise ValueError("input and labels must have the same shape "
                         "(excepting dimensions with width 1)")

    if index is None:
        return func(input[labels > 0], positions[labels > 0])

    index = np.atleast_1d(index)
    if np.any(index.astype(labels.dtype).astype(index.dtype) != index):
        raise ValueError("Cannot convert index values from <%s> to <%s> "
                         "(labels' type) without loss of precision" %
                         (index.dtype, labels.dtype))

    index = index.astype(labels.dtype)

    # optimization: find min/max in index, and select those parts of labels, input, and positions
    lo = index.min()
    hi = index.max()
    mask = (labels >= lo) & (labels <= hi)

    # this also ravels the arrays
    labels = labels[mask]
    input = input[mask]
    positions = positions[mask]

    # sort everything by labels
    label_order = labels.argsort()
    labels = labels[label_order]
    input = input[label_order]
    positions = positions[label_order]

    index_order = index.argsort()
    sorted_index = index[index_order]

    def do_map(inputs, output):
        """labels must be sorted"""
        nidx = sorted_index.size

        # Find boundaries for each stretch of constant labels
        # This could be faster, but we already paid N log N to sort labels.
        lo = np.searchsorted(labels, sorted_index, side='left')
        hi = np.searchsorted(labels, sorted_index, side='right')

        for i, l, h in zip(range(nidx), lo, hi):
            if l == h:
                continue
            output[i] = func(*[inp[l:h] for inp in inputs])

    temp = np.empty(index.shape + output_shape, float)
    temp[:] = -1
    do_map([input, positions], temp)

    output = np.zeros(index.shape + output_shape, float)
    output[index_order] = temp
    if as_scalar:
        output = output[0]

    return output


def largest_connected_component_without_holes(binary_image):
    binary_image = scipy.ndimage.morphology.binary_fill_holes(binary_image)
    labels, m = scipy.ndimage.label(binary_image)
    n = scipy.ndimage.measurements.sum(binary_image, labels, range(m + 1))
    binary_image[labels != np.argmax(n)] = 0
    return binary_image


def density2points(density, regions, fit_gaussians=False):
    X, Y = np.indices(regions.shape)
    X = np.vstack([X.flatten(), Y.flatten()]).T

    labelled, n = scipy.ndimage.label(regions)
    num_instances = scipy.ndimage.sum(density, labelled, range(0, n + 1))
    num_instances = np.round(1.1 * num_instances / np.median(num_instances)).astype(int)
    unique_num_instances = np.unique(num_instances[1:])

    if fit_gaussians:
        positions = []
        k = 1
    else:
        positions = [np.array(scipy.ndimage.center_of_mass(density, labelled, np.where(num_instances == 1)[0]))]
        k = 2

    for i in unique_num_instances[unique_num_instances >= k]:
        if i == 1:
            def func(weights, region):
                model = MultivariateGaussianDistribution.from_samples(X[region], weights=weights)
                return model.parameters[0]
        else:
            def func(weights, region):
                model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, i,
                                                         X[region], weights=weights)
                points = np.zeros((i, 2))
                for j, distribution in enumerate(model.distributions):
                    points[j] = distribution.parameters[0]
                return points

        positions += [labeled_comprehension(density, labelled, np.where(num_instances == i)[0], func,
                                            output_shape=(i, 2)).reshape(-1, 2)]

    return np.vstack(positions)
