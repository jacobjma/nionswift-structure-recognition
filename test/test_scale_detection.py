import pytest

from nionswift_plugin.nionswift_structure_recognition.scale import ScaleDetectionModule, presets
import numpy as np
import torch

@pytest.fixture
def scale_detection_module(ui_column):
    ui, document_controller, column = ui_column
    sdm = ScaleDetectionModule(ui, document_controller)
    sdm.create_widgets(column)
    sdm.set_preset('graphene')
    sdm.fetch_parameters()
    return sdm


def test_scale_detection(scale_detection_module, noisy_image_1, noisy_image_2, noisy_image_3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    noisy_image_1 = torch.tensor(noisy_image_1).to(device)
    noisy_image_2 = torch.tensor(noisy_image_2).to(device)
    noisy_image_3 = torch.tensor(noisy_image_3).to(device)

    sampling = scale_detection_module.detect_scale(noisy_image_1)
    assert np.isclose(sampling, 40 / 512, rtol=.05)

    sampling = scale_detection_module.detect_scale(noisy_image_2)
    assert np.isclose(sampling, 20 / 512, rtol=.1)

    sampling = scale_detection_module.detect_scale(noisy_image_3)
    assert np.isclose(sampling, 10 / 512, rtol=.2)