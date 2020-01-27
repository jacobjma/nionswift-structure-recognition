import numpy as np
import pytest
from abtem.learn.dataset import gaussian_marker_labels
from abtem.learn.structures import graphene_like, random_add_contamination
from abtem.points import fill_rectangle, rotate
from nion.swift import Application
from nion.swift import DocumentController
from nion.swift.Facade import UserInterface
from nion.swift.model import DocumentModel
from nion.ui import TestUI


def create_test_image(gpts, extent, background=1.):
    np.random.seed(7)
    points = graphene_like(a=2.46, labels=[1, 1])
    points = rotate(points, np.random.rand() * 360., rotate_cell=True)
    points = fill_rectangle(points, origin=[0, 0], extent=extent, margin=4)
    points = random_add_contamination(points, 3, extent, .8)
    image = gaussian_marker_labels(points, .4, gpts).astype(np.float32)
    return np.random.poisson(background + image)


@pytest.fixture(scope="session", autouse=True)
def noisy_image_1():
    try:
        image = np.load('noisy_image_1.npy')
    except:
        image = create_test_image(512, 40, 20)
        np.save('noisy_image_1.npy', image)

    return image


@pytest.fixture(scope="session", autouse=True)
def noisy_image_2():
    try:
        image = np.load('noisy_image_2.npy')
    except:
        image = create_test_image(512, 20, 10)
        np.save('noisy_image_2.npy', image)

    return image


@pytest.fixture(scope="session", autouse=True)
def noisy_image_3():
    try:
        image = np.load('noisy_image_3.npy')
    except:
        image = create_test_image(512, 10, 2.5)
        np.save('noisy_image_3.npy', image)

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()
    return image


@pytest.fixture(scope="session", autouse=True)
def ui_column():
    app = Application.Application(TestUI.UserInterface(), set_global=False)
    ui = UserInterface('1', app.ui)
    document_model = DocumentModel.DocumentModel()
    document_controller = DocumentController.DocumentController(app.ui, document_model)
    column = ui.create_column_widget()
    return ui, document_controller, column
