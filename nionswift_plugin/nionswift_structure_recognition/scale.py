import functools
from bisect import bisect_left

import numpy as np
import torch

from .utils import StructureRecognitionModule
from .widgets import Section, line_edit_template, combo_box_template

presets = {'graphene':
               {'crystal_system': 'hexagonal',
                'lattice_constant': 2.46,
                'min_sampling': 0.015,
                }
           }





class ScaleDetectionModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

        self.crystal_system = None
        self.lattice_constant = None

    def create_widgets(self, column):
        section = Section(self.ui, 'Scale detection')
        column.add(section)

        lattice_constant_row, self.lattice_constant_line_edit = line_edit_template(self.ui, 'Lattice constant [Å]')
        min_sampling_row, self.min_sampling_line_edit = line_edit_template(self.ui, 'Min. sampling [Å / pixel]')
        crystal_system_row, self.crystal_system_combo_box = combo_box_template(self.ui, 'Crystal system', ['Hexagonal'])

        section.column.add(crystal_system_row)
        section.column.add(lattice_constant_row)
        section.column.add(min_sampling_row)

    def set_preset(self, name):
        self.crystal_system_combo_box.current_item = presets[name]['crystal_system']
        self.lattice_constant_line_edit.text = presets[name]['lattice_constant']
        self.min_sampling_line_edit.text = presets[name]['min_sampling']

    def fetch_parameters(self):
        self.crystal_system = self.crystal_system_combo_box._widget.current_item.lower()
        self.lattice_constant = float(self.lattice_constant_line_edit._widget.text)
        self.min_sampling = float(self.min_sampling_line_edit._widget.text)

    def detect_scale(self, data):
        if self.crystal_system not in ['hexagonal']:
            raise RuntimeError('structure {} not recognized for scale recognition'.format(self.crystal_system))

        scale = find_hexagonal_sampling(data, a=self.lattice_constant, min_sampling=self.min_sampling)
        return scale
