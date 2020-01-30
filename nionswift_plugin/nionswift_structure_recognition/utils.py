import numpy as np




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
