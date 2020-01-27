import pytest

import torch
import numpy as np


def test_reshape(deep_learning_module, test_data_1):
    preprocessed = deep_learning_module.reshape_images(test_data_1['image'])
    assert len(preprocessed.shape) == 4


def test_rescale(deep_learning_module, scale_detection_module, test_data_2):
    preprocessed = deep_learning_module.reshape_images(test_data_2['image'])
    preprocessed = deep_learning_module.rescale_images(preprocessed, test_data_2['sampling'])

    assert preprocessed.shape[-2] / 16 == preprocessed.shape[-2] // 16
    assert preprocessed.shape[-1] / 16 == preprocessed.shape[-1] // 16
    assert np.isclose(scale_detection_module.detect_scale(preprocessed), 30 / 512, rtol=.05)


def test_normalize(deep_learning_module, test_data_1):
    preprocessed = deep_learning_module.reshape_images(test_data_1['image'])
    preprocessed = deep_learning_module.normalize_images(preprocessed)

    assert np.all(np.isclose(torch.mean(preprocessed, dim=(-1, -2)).cpu().numpy(), 0., atol=1e-6))
    assert np.all(np.isclose(torch.std(preprocessed, dim=(-1, -2)).cpu().numpy(), 1., atol=1e-6))


def test_load_model(deep_learning_module):
    deep_learning_module.load_model()


def test_pipeline(scale_detection_module, deep_learning_module, test_data_1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = torch.tensor(test_data_1['image']).to(device)

    sampling = scale_detection_module.detect_scale(images)

    images = deep_learning_module.reshape_images(images)
    images = deep_learning_module.rescale_images(images, sampling)
    images = deep_learning_module.normalize_images(images)

    mask = deep_learning_module.mask_model(images)
    mask = torch.sum(mask[:, :-1], dim=1)[:, None]

    images = deep_learning_module.normalize_images(images, mask)
    density = deep_learning_module.density_model(images)
    density = mask * density
    density = density.detach().cpu().numpy()
    points = deep_learning_module.nms(density)

    # import matplotlib.pyplot as plt
    # plt.imshow(density[0,0].T)
    # plt.plot(*points.T,'ro')
    # plt.show()
