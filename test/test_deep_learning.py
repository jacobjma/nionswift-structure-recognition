import pytest

import torch
import numpy as np


def test_reshape(deep_learning_module, noisy_image_1):
    preprocessed = deep_learning_module.reshape_images(noisy_image_1)
    assert len(preprocessed.shape) == 4


def test_rescale(deep_learning_module, scale_detection_module, noisy_image_1):
    preprocessed = deep_learning_module.reshape_images(noisy_image_1)
    preprocessed = deep_learning_module.rescale_images(preprocessed, 40 / 512)

    assert preprocessed.shape[-2] / 16 == preprocessed.shape[-2] // 16
    assert preprocessed.shape[-1] / 16 == preprocessed.shape[-1] // 16
    assert np.isclose(scale_detection_module.detect_scale(preprocessed), 30 / 512, rtol=.05)


def test_normalize(deep_learning_module, noisy_image_1):
    preprocessed = deep_learning_module.reshape_images(noisy_image_1)
    preprocessed = deep_learning_module.normalize_images(preprocessed)

    assert np.all(np.isclose(torch.mean(preprocessed, dim=(-1, -2)).cpu().numpy(), 0., atol=1e-6))
    assert np.all(np.isclose(torch.std(preprocessed, dim=(-1, -2)).cpu().numpy(), 1., atol=1e-6))


def test_load_model(deep_learning_module):
    deep_learning_module.load_model()