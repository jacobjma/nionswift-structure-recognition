from nionswift_plugin.nionswift_structure_recognition.model import build_model_from_dict, presets


def test_build(test_data_1):
    model = build_model_from_dict(presets['graphene'])
    points = model.predict(test_data_1['image'])

    #print(test_data_1['image'])
    print(points[0])
    import matplotlib.pyplot as plt
    plt.plot(*points.T, 'o')
    plt.imshow(test_data_1['image'].detach().cpu())
    plt.show()

    #print(output)
