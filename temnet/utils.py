import json
import os

import h5py
import numpy as np
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from skimage.morphology import disk, binary_dilation


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


def is_position_inside_image(positions, shape, margin=0):
    mask = ((positions[:, 0] >= -margin) & (positions[:, 1] >= -margin) &
            (positions[:, 0] < shape[0] + margin) & (positions[:, 1] < shape[1] + margin))
    return mask


def label_to_index_generator(labels, first_label=0):
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(first_label, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield indices[l:h]


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


def walk_dir(path, ending):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file[-len(ending):] == ending:
                files.append(os.path.join(r, file))

    return files


def insert_folder_in_path(base_path, folder_name, full_path):
    folder_path, fname = os.path.split(os.path.relpath(full_path, base_path))
    path = os.path.join(os.path.join(base_path, folder_name), folder_path, fname)
    return path


def as_rgb_image(array, normalize=True):
    if normalize:
        array = ((array - array.min()) / array.ptp() * 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8)

    return np.tile(array[..., None], 3)


def draw_points(image, points, radius, color):
    if len(points) == 0:
        return image

    shape = (image.shape[0] + 2 * radius, image.shape[1] + 2 * radius)
    cols, rows = np.round(points + radius).astype(np.int).T
    inside = (cols > 0) & (cols < shape[0]) & (rows > 0) & (rows < shape[1])
    cols = cols[inside]
    rows = rows[inside]
    mask = np.zeros(shape, dtype=np.bool)
    mask[rows, cols] = 1
    mask = binary_dilation(mask, disk(radius))
    mask = mask[radius:-radius, radius:-radius]
    image[mask] = color
    return image


def read_nion_ndata(path):
    data_item = np.load(path)
    description = json.loads(data_item['metadata.json'])
    data = data_item['data']
    return data, description


def read_nion_h5(path):
    f = h5py.File(path, 'r')
    return f['data'][:]


def subdivide_into_batches(num_items: int, num_batches: int = None, max_batch: int = None):
    """
    Split an n integer into m (almost) equal integers, such that the sum of smaller integers equals n.

    Parameters
    ----------
    n: int
        The integer to split.
    m: int
        The number integers n will be split into.

    Returns
    -------
    list of int
    """
    if (num_batches is not None) & (max_batch is not None):
        raise RuntimeError()

    if num_batches is None:
        if max_batch is not None:
            num_batches = (num_items + (-num_items % max_batch)) // max_batch
        else:
            raise RuntimeError()

    if num_items < num_batches:
        raise RuntimeError('num_batches may not be larger than num_items')

    elif num_items % num_batches == 0:
        return [num_items // num_batches] * num_batches
    else:
        v = []
        zp = num_batches - (num_items % num_batches)
        pp = num_items // num_batches
        for i in range(num_batches):
            if i >= zp:
                v = [pp + 1] + v
            else:
                v = [pp] + v
        return v


def generate_batches(num_items: int, num_batches: int = None, max_batch: int = None, start=0):
    for batch in subdivide_into_batches(num_items, num_batches, max_batch):
        end = start + batch
        yield start, end

        start = end
