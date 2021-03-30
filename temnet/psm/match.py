import numpy as np
import scipy.spatial as spatial
import torch
import torch.fft

from temnet.psm.segments import stack, SegmentedPoints


def superpose_gaussians(positions, shape, sigma, *args):
    d = ()
    for arg in args:
        d += (max(arg).item() + 1,)

    array = torch.zeros(d + shape)

    rounded = torch.floor(positions).type(torch.long)
    rows, cols = rounded[:, 0], rounded[:, 1]

    kx = torch.tensor(np.fft.fftfreq(shape[0]))
    ky = torch.tensor(np.fft.fftfreq(shape[1]))
    k = torch.sqrt(kx[:, None] ** 2 + ky[None] ** 2)
    g = torch.exp(-k ** 2 * sigma ** 2 * 2 * np.pi ** 2) * sigma ** 2 * 2 * np.pi

    array[args + (rows, cols)] += (1 - (positions[:, 0] - rows)) * (1 - (positions[:, 1] - cols))
    array[args + ((rows + 1) % shape[0], cols)] += (positions[:, 0] - rows) * (1 - (positions[:, 1] - cols))
    array[args + (rows, (cols + 1) % shape[1])] += (1 - (positions[:, 0] - rows)) * (positions[:, 1] - cols)
    array[args + ((rows + 1) % shape[0], (cols + 1) % shape[1])] += (rows - positions[:, 0]) * (cols - positions[:, 1])

    array = torch.fft.ifftn(torch.fft.fftn(array, dim=(-2, -1)) * g, dim=(-2, -1)).real
    return array


# def gaussian_superposition_from_segments(segments, n, sigma, window=None):
#     stacked = stack(segments)
#     points = torch.tensor(stacked.points)
#     labels = torch.tensor(stacked.labels, dtype=torch.long)
#
#     if window is None:
#         window = (points.max() - points.min()) + 4 * sigma
#     points += window / 2
#     points *= n / window
#
#     sizes = [len(segment) for segment in segments]
#     segment_labels = torch.tensor(np.repeat(np.arange(len(segments)), sizes), dtype=torch.long)
#     return superpose_gaussians(points, (n,) * 2, sigma, segment_labels, labels)


def gaussian_overlap(segments1, segments2, sigma, n, window):
    gaussians = []

    for segments in (segments1, segments2):
        stacked = stack(segments)
        points = torch.tensor(stacked.points)
        labels = torch.tensor(stacked.labels, dtype=torch.long)

        # if window is None:
        #    window = (points.max() - points.min()) + 4 * sigma
        points += window / 2
        points *= n / window

        sizes = [len(segment) for segment in segments]
        segment_labels = torch.tensor(np.repeat(np.arange(len(segments)), sizes), dtype=torch.long)
        gaussians += [superpose_gaussians(points, (n,) * 2, sigma, segment_labels, labels)]

    overlaps = (gaussians[0][None] * gaussians[1][:, None]).sum(-3)#.sum((-3, -2, -1))
    # import matplotlib.pyplot as plt
    # plt.imshow(overlaps[35,50])
    # plt.show()
    # sss


    # plt.imshow(gaussians[0][50,0])
    # plt.show()
    # plt.imshow(gaussians[1][0,0])
    # plt.show()
    # ss
    return (gaussians[0][None] * gaussians[1][:, None]).sum((-3, -2, -1))


def order_as_closest(templates, segments):
    extracted_segments = segments.extract(True)

    new_segments = []
    for i, (indices, extracted, template) in enumerate(zip(segments.segments, extracted_segments, templates)):
        point_tree = spatial.cKDTree(extracted.points)

        nearest = point_tree.query(template.points, 1)[1]
        new_segments += [indices[nearest]]

    return SegmentedPoints(segments.points, new_segments)
