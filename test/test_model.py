from nionswift_plugin.nionswift_structure_recognition.model import build_model_from_dict, presets
from nionswift_plugin.nionswift_structure_recognition.visualization import create_visualization

def test_build(test_data_1):
    model = build_model_from_dict(presets['graphene'])

    images = test_data_1['image'][None]

    points = model.predict(images)

    visualization = create_visualization(images, None, None, points, presets['graphene']['visualization'])




    #print(test_data_1['image'])
    #print(points[0])
    import matplotlib.pyplot as plt
    plt.imshow(visualization[0])
    plt.show()

    #plt.plot(*points.T, 'o')
    #plt.imshow(test_data_1['image'].detach().cpu())
    #plt.show()

    #print(output)
