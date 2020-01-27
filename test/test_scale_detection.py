import numpy as np


def test_scale_detection(scale_detection_module, noisy_image_1, noisy_image_2, noisy_image_3):
    sampling = scale_detection_module.detect_scale(noisy_image_1)
    assert np.isclose(sampling, 40 / 512, rtol=.05)

    sampling = scale_detection_module.detect_scale(noisy_image_2)
    assert np.isclose(sampling, 20 / 512, rtol=.1)

    sampling = scale_detection_module.detect_scale(noisy_image_3)
    assert np.isclose(sampling, 10 / 512, rtol=.2)
