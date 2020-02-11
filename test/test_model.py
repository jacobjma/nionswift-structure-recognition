from nionswift_plugin.nionswift_structure_recognition.model import build_model_from_dict, presets
from nionswift_plugin.nionswift_structure_recognition.visualization import create_visualization
import numpy as np


def test_build():
    model = build_model_from_dict(presets['graphene'])

    images = np.load('test_image.npy')
    images = np.random.poisson(1 + images).astype(np.float32)
    output = model.predict(images)

    points = output['points'][0]
    probabilities = output['probabilities'][0]


    import matplotlib.pyplot as plt
    plt.imshow(images, cmap='gray')
    plt.scatter(*points.T, c=probabilities[:, 1], s=5, cmap='autumn')
    plt.show()

    # visualization = create_visualization(images, None, None, points, presets['graphene']['visualization'])

    # print(test_data_1['image'])
    # print(points[0])
    # import matplotlib.pyplot as plt
    # plt.imshow(visualization[0])
    # plt.show()

    # plt.plot(*points.T, 'o')
    # plt.imshow(test_data_1['image'].detach().cpu())
    # plt.show()

    # print(output)
