import numpy as np
import pytest
import torch

from nion.swift import Application
from nion.swift import DocumentController
from nion.swift.Facade import UserInterface
from nion.swift.model import DocumentModel
from nion.ui import TestUI
import os
# from nionswift_plugin.nionswift_structure_recognition.dl import DeepLearningModule
# from nionswift_plugin.nionswift_structure_recognition.scale import ScaleDetectionModule
# from nionswift_plugin.nionswift_structure_recognition.visualization import VisualizationModule


def create_data(extent, gpts, background=1.):
    from abtem.learn.dataset import gaussian_marker_labels
    from abtem.learn.structures import graphene_like, random_add_contamination
    from abtem.points import fill_rectangle, rotate
    np.random.seed(7)
    points = graphene_like(a=2.46, labels=[0, 0])
    points = rotate(points, np.random.rand() * 360., rotate_cell=True)
    points = fill_rectangle(points, origin=[0, 0], extent=extent, margin=4)
    points = random_add_contamination(points, 1, np.diag(points.cell), .8)
    density = gaussian_marker_labels(points, .4, gpts).astype(np.float32)
    image = np.random.poisson(background + density).astype(np.float32)
    sampling = extent / gpts
    return {'image': image, 'density': density, 'positions': points.positions / sampling, 'sampling': extent / gpts}


@pytest.fixture(scope="session", autouse=True)
def test_data_1():
    path = os.path.join(os.path.dirname(__file__), 'test_data_1.npz')
    try:
        npzfile = np.load(path)
        data = {key: npzfile[key] for key in npzfile.keys()}
    except:
        data = create_data(40, 512, 2)
        np.savez(path, **data)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data['image'] = torch.tensor(data['image']).to(device)
    data['density'] = torch.tensor(data['density']).to(device)
    return data

#
# @pytest.fixture(scope="session", autouse=True)
# def test_data_2():
#     path = os.path.join(os.path.dirname(__file__), 'test_data_1.npz')
#     try:
#         npzfile = np.load(path)
#         data = {key: npzfile[key] for key in npzfile.keys()}
#     except:
#         data = create_data(20, 512, 2)
#         np.savez(path, **data)
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     data['image'] = torch.tensor(data['image']).to(device)
#     data['density'] = torch.tensor(data['density']).to(device)
#     return data
#
#
# @pytest.fixture(scope="session", autouse=True)
# def ui_column():
#     app = Application.Application(TestUI.UserInterface(), set_global=False)
#     ui = UserInterface('1', app.ui)
#     document_model = DocumentModel.DocumentModel()
#     document_controller = DocumentController.DocumentController(app.ui, document_model)
#     column = ui.create_column_widget()
#     return ui, document_controller, column
#
#
# @pytest.fixture
# def scale_detection_module(ui_column):
#     ui, document_controller, column = ui_column
#     sdm = ScaleDetectionModule(ui, document_controller)
#     sdm.create_widgets(column)
#     sdm.set_preset('graphene')
#     sdm.fetch_parameters()
#     return sdm
#
#
# @pytest.fixture
# def deep_learning_module(ui_column):
#     ui, document_controller, column = ui_column
#     sdm = DeepLearningModule(ui, document_controller)
#     sdm.create_widgets(column)
#     sdm.set_preset('graphene')
#     sdm.fetch_parameters()
#     return sdm
#
#
# @pytest.fixture
# def visualization_module(ui_column):
#     ui, document_controller, column = ui_column
#     sdm = VisualizationModule(ui, document_controller)
#     sdm.create_widgets(column)
#     sdm.set_preset('graphene')
#     sdm.fetch_parameters()
#     return sdm
