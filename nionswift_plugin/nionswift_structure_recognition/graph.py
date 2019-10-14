import numpy as np

from .psm.graph import GeometricGraph
from .psm.rmsd import RMSD
from .psm.templates import regular_polygon
from .utils import StructureRecognitionModule
from .widgets import Section, line_edit_template


class GraphModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

        self.alpha = None

    def create_widgets(self, column):
        section = Section(self.ui, 'Graph')
        column.add(section)

        alpha_row, self.alpha_line_edit = line_edit_template(self.ui, 'alpha', default_text=1,
                                                             placeholder_text='Do not build')

        section.column.add(alpha_row)

        cutoff_row, self.cutoff_line_edit = line_edit_template(self.ui, 'cutoff', default_text='',
                                                               placeholder_text='Do not use')

        section.column.add(cutoff_row)

    def set_preset(self, name):
        pass

    def fetch_parameters(self):
        if len(self.alpha_line_edit.text) > 0:
            self.alpha = float(self.alpha_line_edit.text)

        else:
            self.alpha = None

        if len(self.cutoff_line_edit.text) > 0:
            self.cutoff = float(self.cutoff_line_edit.text)

        else:
            self.cutoff = np.inf

    def build_graph(self, points, sampling):

        if self.alpha is not None:

            graph = GeometricGraph(points)

            graph.build_stable_delaunay_graph(self.alpha, self.cutoff / sampling)

            return graph

        else:
            return None

    def register_faces(self, graph):
        # templates = [regular_polygon(1,i) for i in range()]
        template = regular_polygon(1, 6)
        rmsd = RMSD().register(graph.points, graph.faces(), [template])
        return rmsd
