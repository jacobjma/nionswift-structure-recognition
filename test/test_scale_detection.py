import numpy as np
from nionswift_plugin.nionswift_structure_recognition.scale import find_hexagonal_sampling
import matplotlib.pyplot as plt

def test_scale_detection(test_data_1):
    print(test_data_1['image'].shape)

    plt.imshow(test_data_1['image'])
    plt.show()

    sampling = find_hexagonal_sampling(test_data_1['image'], 2.46, .01, .1)
    print(sampling)

    #sampling = scale_detection_module.detect_scale(test_data_1['image'])
    #assert np.isclose(sampling, test_data_1['sampling'], rtol=.05)

    #sampling = scale_detection_module.detect_scale(test_data_2['image'])
    #assert np.isclose(sampling, test_data_2['sampling'], rtol=.1)
