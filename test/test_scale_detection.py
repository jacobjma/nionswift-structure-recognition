import numpy as np


def test_scale_detection(scale_detection_module, test_data_1, test_data_2):
    sampling = scale_detection_module.detect_scale(test_data_1['image'])
    assert np.isclose(sampling, test_data_1['sampling'], rtol=.05)

    sampling = scale_detection_module.detect_scale(test_data_2['image'])
    assert np.isclose(sampling, test_data_2['sampling'], rtol=.1)
