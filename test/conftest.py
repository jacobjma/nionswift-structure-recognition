import numpy as np
import pytest
import torch
from abtem.learn.dataset import gaussian_marker_labels
from abtem.learn.structures import graphene_like, random_add_contamination
from abtem.points import fill_rectangle, rotate
from nion.swift import Application
from nion.swift import DocumentController
from nion.swift.Facade import UserInterface
from nion.swift.model import DocumentModel
from nion.ui import TestUI

from nionswift_plugin.nionswift_structure_recognition.dl import DeepLearningModule
from nionswift_plugin.nionswift_structure_recognition.scale import ScaleDetectionModule


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
        image = create_test_image(512, 40, 20).astype(np.float32)
        np.save('noisy_image_1.npy', image)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = torch.tensor(image).to(device)
    return image


@pytest.fixture(scope="session", autouse=True)
def noisy_image_2():
    try:
        image = np.load('noisy_image_2.npy')
    except:
        image = create_test_image(512, 20, 10).astype(np.float32)
        np.save('noisy_image_2.npy', image)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = torch.tensor(image).to(device)
    return image


@pytest.fixture(scope="session", autouse=True)
def noisy_image_3():
    try:
        image = np.load('noisy_image_3.npy')
    except:
        image = create_test_image(512, 10, 2.5).astype(np.float32)
        np.save('noisy_image_3.npy', image)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = torch.tensor(image).to(device)
    return image


@pytest.fixture(scope="session", autouse=True)
def ui_column():
    app = Application.Application(TestUI.UserInterface(), set_global=False)
    ui = UserInterface('1', app.ui)
    document_model = DocumentModel.DocumentModel()
    document_controller = DocumentController.DocumentController(app.ui, document_model)
    column = ui.create_column_widget()
    return ui, document_controller, column


@pytest.fixture
def scale_detection_module(ui_column):
    ui, document_controller, column = ui_column
    sdm = ScaleDetectionModule(ui, document_controller)
    sdm.create_widgets(column)
    sdm.set_preset('graphene')
    sdm.fetch_parameters()
    return sdm


@pytest.fixture
def deep_learning_module(ui_column):
    ui, document_controller, column = ui_column
    sdm = DeepLearningModule(ui, document_controller)
    sdm.create_widgets(column)
    sdm.set_preset('graphene')
    sdm.fetch_parameters()
    return sdm
