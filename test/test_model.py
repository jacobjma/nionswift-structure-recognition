from nionswift_plugin.nionswift_structure_recognition.model import build_model_from_dict, presets
from nionswift_plugin.nionswift_structure_recognition.visualization import create_visualization
import numpy as np
from skimage.io import imread
from scipy.ndimage import gaussian_filter

def test_build():
    model = build_model_from_dict(presets['graphene'])

    # images = np.load('test_image.npy')

    #image = imread('0074_0.00025198_0.00027884.tiff')
    #image = imread('0041_0.00025198_0.00027887.tiff')
    image = imread('0035_9.91682e-05_0.000396419.tiff')

    # images = np.random.poisson(1 + images).astype(np.float32)
    output = model.predict(image[None])

    points = output['points'][0]
    # probabilities = output['probabilities'][0]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap='gray')
    plt.scatter(*points.T, color='r')
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
