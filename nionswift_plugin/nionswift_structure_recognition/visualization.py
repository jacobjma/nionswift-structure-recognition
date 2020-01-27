import cv2
import numpy as np
from matplotlib import colors as mcolors
import matplotlib
from .utils import StructureRecognitionModule
from .widgets import Section, combo_box_template, check_box_template, line_edit_template

named_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


def get_colors_from_cmap(c, cmap=None, vmin=None, vmax=None):
    if cmap is None:
        cmap = matplotlib.cm.get_cmap('viridis')

    elif isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    if vmin is None:
        vmin = np.nanmin(c)

    if vmax is None:
        vmax = np.nanmax(c)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    c = np.array(c, dtype=float)

    valid = np.isnan(c) == 0
    colors = np.zeros((len(c), 4))
    colors[valid] = cmap(norm(c[valid]))

    return colors


def add_faces(points, faces, image, colors):
    points = np.round(points).astype(int)

    for face, color in zip(faces, colors):
        cv2.fillConvexPoly(image, points[face][:, ::-1], tuple(map(int, color)))

    return image


def add_edges(points, edges, image, color, thickness=1):
    points = np.round(points).astype(int)
    for edge in edges:
        cv2.line(image, tuple(points[edge[0]][::-1]), tuple(points[edge[1]][::-1]), color=color,
                 thickness=thickness)

    return image


def add_points(points, image, size, colors):
    points = np.round(points).astype(np.int)

    for point, color in zip(points, colors):
        cv2.circle(image, (point[1], point[0]), size, tuple(map(int, color)), -1)

    return image


# class GraphicOptions(object):
#
#     def __init__(self, graphic_check_box, contents_column):
#         self.graphic_check_box = graphic_check_box
#         self.column = ui.cre
#         self.contents_column = contents_column
#
#     def check_box_changed(self):
#         self.contents_column._widget.remove_all()


class VisualizationModule(StructureRecognitionModule):

    def __init__(self, ui, document_controller):
        super().__init__(ui, document_controller)

        self.training_sampling = None

    def create_widgets(self, column):
        section = Section(self.ui, 'Visualization')
        column.add(section)

        background_row, self.background_combo_box = combo_box_template(self.ui, 'Background',
                                                                       ['Image', 'Density', 'Classes', 'Confidence',
                                                                        'Solid'])
        section.column.add(background_row)

        points_row, self.points_check_box = check_box_template(self.ui, 'Overlay points')
        section.column.add(points_row)

        points_settings_column = self.ui.create_column_widget()
        points_color_column = self.ui.create_column_widget()

        self.points_size_line_edit = None
        self.points_color_combo_box = None
        self.point_color_solid_line_edit = None

        section.column.add(points_settings_column)
        section.column.add(points_color_column)
        section.column.add_spacing(5)

        def points_check_box_changed(checked):

            points_settings_column._widget.remove_all()
            points_color_column._widget.remove_all()

            def point_color_combo_box_changed(item):
                points_color_column._widget.remove_all()

                if item.lower() == 'solid':
                    points_color_solid_row, self.point_color_solid_line_edit = line_edit_template(self.ui, 'Color',
                                                                                                  default_text='red')
                    points_color_column.add(points_color_solid_row)

            if checked:
                points_size_row, self.points_size_line_edit = line_edit_template(self.ui, 'Point size', default_text=3)

                points_color_row, self.points_color_combo_box = combo_box_template(self.ui, 'Point color',
                                                                                   ['Class', 'Solid'])

                self.points_color_combo_box.on_current_item_changed = point_color_combo_box_changed

                points_settings_column.add_spacing(5)
                points_settings_column.add(points_size_row)
                points_settings_column.add(points_color_row)

                point_color_combo_box_changed(self.points_color_combo_box.current_item)

            else:
                self.points_size_line_edit = None

        self.points_check_box.on_checked_changed = points_check_box_changed
        self.points_check_box.checked = True
        points_check_box_changed(self.points_check_box.checked)

        graph_row, self.graph_check_box = check_box_template(self.ui, 'Overlay graph')
        section.column.add(graph_row)

        graph_settings_column = self.ui.create_column_widget()
        self.line_width_line_edit = None
        section.column.add(graph_settings_column)
        section.column.add_spacing(5)

        def graph_check_box_changed(checked):
            graph_settings_column._widget.remove_all()

            if checked:
                line_width_row, self.line_width_line_edit = line_edit_template(self.ui, 'Line width', default_text=2)

                graph_settings_column.add_spacing(5)
                graph_settings_column.add(line_width_row)

        self.graph_check_box.on_checked_changed = graph_check_box_changed
        self.graph_check_box.checked = True
        graph_check_box_changed(self.graph_check_box.checked)

        faces_row, self.faces_check_box = check_box_template(self.ui, 'Overlay faces')
        section.column.add(faces_row)

        faces_settings_column = self.ui.create_column_widget()
        self.faces_color_combo_box = None
        self.faces_cmap_combo_box = None
        section.column.add(faces_settings_column)
        section.column.add_spacing(5)

        def faces_check_box_changed(checked):
            faces_settings_column._widget.remove_all()

            if checked:
                faces_color_row, self.faces_color_combo_box = combo_box_template(self.ui, 'Face color',
                                                                                 ['Size', 'RMSD', 'exx', 'eyy'])

                faces_settings_column.add_spacing(5)
                faces_settings_column.add(faces_color_row)

                faces_cmap_row, self.faces_cmap_combo_box = combo_box_template(self.ui, 'Color map',
                                                                               ['gray', 'viridis', 'plasma', 'Paired',
                                                                                'tab10'])

                faces_settings_column.add_spacing(5)
                faces_settings_column.add(faces_cmap_row)

                # faces_vmin_row, self.faces_vmin_line_edit = line_edit_template(self.ui, 'vmin', default_text=0)
                # faces_vmax_row, self.faces_vmax_line_edit = line_edit_template(self.ui, 'vmin', default_text=0)

        self.faces_check_box.on_checked_changed = faces_check_box_changed
        self.faces_check_box.checked = False
        faces_check_box_changed(self.faces_check_box.checked)

    def set_preset(self, x):
        pass

    def fetch_parameters(self):
        self.background = self.background_combo_box._widget.current_item.lower()
        self.overlay_points = self.points_check_box.checked

        if self.overlay_points:
            self.points_size = int(self.points_size_line_edit.text)

            self.points_color = self.points_color_combo_box.current_item.lower()

            if self.points_color == 'solid':
                self.points_color_solid = self.point_color_solid_line_edit.text

        self.overlay_graph = self.graph_check_box.checked

        if self.overlay_graph:
            self.line_width = int(self.line_width_line_edit.text)

        self.overlay_faces = self.faces_check_box.checked

        if self.overlay_faces:
            self.faces_color = self.faces_color_combo_box.current_item.lower()
            self.faces_cmap = self.faces_cmap_combo_box.current_item


    def create_background(self, image, density):

        if self.background == 'image':
            visualization = ((image - image.min()) / image.ptp() * 255).astype(np.uint8)
            visualization = np.tile(visualization[..., None], (1, 1, 3))

        elif self.background == 'density':
            visualization = (density * 255).astype(np.uint8)
            visualization = np.tile(visualization[..., None], (1, 1, 3))

        # elif self.background == 'classes':
        #     visualization = (classes / 3 * 255).astype(np.uint8)
        #     visualization = np.tile(visualization[..., None], (1, 1, 3))

        elif self.background == 'solid':
            visualization = None

        else:
            raise RuntimeError()

        return visualization

    # def create_visualization(self, image, density, points):
    #
    #     # canvas = (0, extent[0], 0, extent[1])
    #     # shape = image.shape
    #     # points = points[:, ::]
    #     # points = scale_points_to_canvas(points, canvas, shape)
    #
    #     if self.background == 'image':
    #         visualization = ((image - image.min()) / image.ptp() * 255).astype(np.uint8)
    #         visualization = np.tile(visualization[..., None], (1, 1, 3))
    #
    #     elif self.background == 'density':
    #         visualization = (density * 255).astype(np.uint8)
    #         visualization = np.tile(visualization[..., None], (1, 1, 3))
    #
    #     # elif self.background == 'classes':
    #     #     visualization = (classes / 3 * 255).astype(np.uint8)
    #     #     visualization = np.tile(visualization[..., None], (1, 1, 3))
    #
    #     elif self.background == 'solid':
    #         visualization = None
    #
    #     else:
    #         raise RuntimeError()
    #
    #     # if self.overlay_faces:
    #     #     if self.faces_color == 'size':
    #     #         colors = graph.faces().sizes()
    #     #         vmin = 0
    #     #         vmax = 10
    #     #
    #     #     elif self.faces_color == 'rmsd':
    #     #         colors = rmsd
    #     #         vmin = 0
    #     #         vmax = np.max(rmsd[rmsd != np.inf])
    #     #
    #     #     else:
    #     #         raise RuntimeError()
    #     #
    #     #     colors = (get_colors_from_cmap(colors, self.faces_cmap, vmin, vmax) * 255).astype(int)
    #     #
    #     #     visualization = add_faces(graph.points, graph.faces()[:-1], visualization, colors)
    #
    #     # if self.overlay_graph:
    #     #     visualization = add_edges(graph.points, graph.edges(), visualization, (0, 0, 0), self.line_width)
    #
    #     if self.overlay_points:
    #         if self.points_color == 'solid':
    #             color = mcolors.to_rgba(named_colors[self.points_color_solid])[:3]
    #             colors = [tuple([int(x * 255) for x in color[::-1]])] * len(points)
    #
    #         # elif self.points_color == 'class':
    #         #     colors = (get_colors_from_cmap(probabilities[:, 2], 'autumn', vmin=0, vmax=1) * 255).astype(int)
    #         #     colors = colors[:, :-1][:, ::-1]
    #
    #         else:
    #             raise NotImplementedError()
    #
    #         visualization = add_points(points, visualization, self.points_size, colors)
    #
    #     return visualization