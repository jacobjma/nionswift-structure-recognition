from nionswift_plugin.nionswift_structure_recognition.model import load_preset_model
from nionswift_plugin.nionswift_structure_recognition.visualization import segmentation_to_uint8_image
import matplotlib.pyplot as plt

def test_build(test_data_1):
    model = load_preset_model('graphene')

    output = model(test_data_1['image'], .05)

    plt.imshow(segmentation_to_uint8_image(output['segmentation']))
    plt.show()
    #points = model.predict(test_data_1['image'])

    #print(test_data_1['image'])
    #print(points[0])
    #import matplotlib.pyplot as plt
    #plt.plot(*points.T, 'o')
    #plt.imshow(test_data_1['image'].detach().cpu())
    #plt.show()

    #print(output)
