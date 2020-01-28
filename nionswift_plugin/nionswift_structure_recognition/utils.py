import numpy as np


def sub2ind(rows, cols, array_shape):
    return rows * array_shape[1] + cols


def ind2sub(array_shape, ind):
    rows = (np.int32(ind) // array_shape[1])
    cols = (np.int32(ind) % array_shape[1])
    return (rows, cols)


class StructureRecognitionModule(object):

    def __init__(self, ui, document_controller):
        self.ui = ui
        self.document_controller = document_controller

    def create_widgets(self, column):
        raise NotImplementedError()

    def set_preset(self, name):
        raise NotImplementedError()

    def fetch_parameters(self):
        raise NotImplementedError()
