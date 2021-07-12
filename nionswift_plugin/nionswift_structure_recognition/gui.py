import gettext
import logging
import threading

import numpy as np
from fourier_scale_calibration import FourierSpaceCalibrator
from nion.utils import Binding, Event

from psm.geometry import regular_polygon
from psm.graph import stable_delaunay_graph
from psm.rmsd import pairwise_rmsd
from psm.select import select_faces_around_nodes, select_nodes_in_faces
from psm.structures.graphene import defect_fingerprint
from psm.utils import flatten_list_of_lists
from .model import load_preset_model
from .scale import RealSpaceCalibrator
from .visualization import add_points, array_to_uint8_image, segmentation_to_uint8_image, add_edges, add_polygons, \
    add_text

_ = gettext.gettext

from nion.ui import Widgets

logger = logging.getLogger(__name__)


def line_edit_template(ui, label, default_text=None, placeholder_text=None):
    row = ui.create_row_widget()
    row.add(ui.create_label_widget(label))
    row.add_spacing(5)
    widget = ui.create_line_edit_widget()
    row.add(widget)
    row.add_spacing(5)
    widget._widget._behavior.placeholder_text = placeholder_text
    widget.text = default_text
    return row, widget


def push_button_template(ui, label, callback=None):
    row = ui.create_row_widget()
    widget = ui.create_push_button_widget(label)
    widget.on_clicked = callback
    row.add(widget)
    return row, widget


def combo_box_template(ui, label, items, indent=False):
    row = ui.create_row_widget()
    if indent:
        row.add_spacing(8)
    row.add(ui.create_label_widget(label))
    row.add_spacing(5)
    widget = ui.create_combo_box_widget(items=items)
    row.add(widget)
    # row.add_stretch()
    return row, widget


def check_box_template(ui, label):
    row = ui.create_row_widget()
    widget = ui.create_check_box_widget(label)
    # row.add_spacing(5)
    row.add(widget)
    return row, widget


class ScrollArea:

    def __init__(self, ui):
        self.__ui = ui
        self.__scroll_area_widget = ui.create_scroll_area_widget()

    @property
    def _ui(self):
        return self.__ui

    @property
    def _widget(self):
        return self.__scroll_area_widget

    @property
    def content(self):
        return self._widget.content

    @content.setter
    def content(self, value):
        self._widget.content = value


class Section:

    def __init__(self, ui, title):
        self.__ui = ui
        self.__section_content_column = self.__ui._ui.create_column_widget()
        self.__section_widget = Widgets.SectionWidget(self.__ui._ui, title, self.__section_content_column, 'test')
        self.column = ui.create_column_widget()
        self.__section_content_column.add(self.column._widget)

    @property
    def _ui(self):
        return self.__ui

    @property
    def _widget(self):
        return self.__section_widget

    @property
    def _section_content_column(self):
        return self.__section_content_column


class DeepLearningSection(Section):

    def __init__(self, ui):
        super().__init__(ui, 'Deep Learning')
        self.property_changed_event = Event.Event()
        model_row, self.model_combo_box = combo_box_template(ui, 'Model', ['Graphene'])
        recurrent_normalize_row, self.recurrent_normalize_check_box = check_box_template(ui, 'Recurrent Normalization')
        self.recurrent_normalize = True
        self.recurrent_normalize_check_box._widget.bind_checked(Binding.PropertyBinding(self, 'recurrent_normalize'))

        boundary_extrapolation_row, self.boundary_extrapolation_line_edit = line_edit_template(self._ui,
                                                                                               'Bondary extrapolation [Å]')
        self.boundary_extrapolation_line_edit._widget._behavior.enabled = False

        self.column.add(model_row)
        self.column.add(recurrent_normalize_row)
        self.column.add(boundary_extrapolation_row)


class ScaleDetectionSection(Section):

    def __init__(self, ui):
        super().__init__(ui, 'Calibrate')
        self.property_changed_event = Event.Event()

        space_row, self.space_combo_box = combo_box_template(self._ui, 'Space', ['Fourier-space', 'Real-space'])
        self.space = 0
        self.space_combo_box._widget.bind_current_index(Binding.PropertyBinding(self, 'space'))

        template_row, self.template_combo_box = combo_box_template(self._ui, 'Template', ['Hexagonal'])
        self.template = 0
        self.template_combo_box._widget.bind_current_index(Binding.PropertyBinding(self, 'template'))

        lattice_constant_row, self.lattice_constant_line_edit = line_edit_template(self._ui, 'Lattice constant [Å]')
        self.lattice_constant = 2.46
        self.lattice_constant_line_edit._widget.bind_text(Binding.PropertyBinding(self, 'lattice_constant'))

        min_sampling_row, self.min_sampling_line_edit = line_edit_template(self._ui, 'Min. sampling [Å / pixel]')
        self.min_sampling = .01
        self.min_sampling_line_edit._widget.bind_text(Binding.PropertyBinding(self, 'min_sampling'))

        max_sampling_row, self.max_sampling_line_edit = line_edit_template(self._ui, 'Max. sampling [Å / pixel]')
        self.max_sampling = .1
        self.max_sampling_line_edit._widget.bind_text(Binding.PropertyBinding(self, 'max_sampling'))

        use_2nd_order_row, self.use_2nd_order_check_box = check_box_template(self._ui, 'Use 2nd order')
        self.use_2nd_order = True
        self.use_2nd_order_check_box._widget.bind_checked(Binding.PropertyBinding(self, 'use_2nd_order'))

        calibrate_next_row, self.calibrate_next_check_box = check_box_template(self._ui, 'Calibrate next acquisition')
        self.calibrate_next = True
        self.calibrate_next_check_box._widget.bind_checked(Binding.PropertyBinding(self, 'calibrate_next'))

        self.column.add(space_row)
        self.column.add(template_row)
        self.column.add(lattice_constant_row)
        self.column.add(min_sampling_row)
        self.column.add(max_sampling_row)
        self.column.add(use_2nd_order_row)
        self.column.add(calibrate_next_row)

    def calibrate(self, image, model):
        lattice_constant = float(self.lattice_constant)
        min_sampling = float(self.min_sampling)
        max_sampling = float(self.max_sampling)

        if self.template == 0:
            template = 'hexagonal'
        else:
            raise RuntimeError()

        if self.space == 0:
            if self.use_2nd_order:
                self.calibrator = FourierSpaceCalibrator(template='2nd-order-hexagonal',
                                                         lattice_constant=lattice_constant,
                                                         min_sampling=min_sampling,
                                                         max_sampling=max_sampling)
            else:
                self.calibrator = FourierSpaceCalibrator(template='hexagonal',
                                                         lattice_constant=lattice_constant,
                                                         min_sampling=min_sampling,
                                                         max_sampling=max_sampling)

        elif self.space == 1:
            self.calibrator = RealSpaceCalibrator(model, template, lattice_constant, min_sampling, max_sampling)

        else:
            raise RuntimeError()

        self.sampling = self.calibrator(image)

        return self.sampling


class GraphSection(Section):

    def __init__(self, ui):
        super().__init__(ui, 'Graph')
        self.property_changed_event = Event.Event()

        alpha_row, self.alpha_line_edit = line_edit_template(ui, 'Alpha [rad.]', default_text=2.)
        self.alpha = 2
        self.alpha_line_edit._widget.bind_text(Binding.PropertyBinding(self, 'alpha'))

        cutoff_row, self.cutoff_line_edit = line_edit_template(ui, 'Cutoff radius [Å]', default_text=5.)
        self.cutoff = 10.
        self.cutoff_line_edit._widget.bind_text(Binding.PropertyBinding(self, 'cutoff'))

        library_row, self.library_combo_box = combo_box_template(ui, 'Template library', ['Graphene'])

        match_row, self.match_combo_box = combo_box_template(ui, 'Match algorithm', ['Heuristic', 'TopoSort', 'CPD'])

        self.column.add(alpha_row)
        self.column.add(cutoff_row)
        self.column.add(library_row)
        self.column.add(match_row)

    def build_graph(self, points, labels, sampling):
        alpha = float(self.alpha)
        cutoff = float(self.cutoff)
        graph = stable_delaunay_graph(points, alpha, cutoff / sampling)
        graph.set_labels(labels)
        return graph

    def analyze_defects(self, graph, sampling):
        dual = graph.dual()

        contamination_faces = select_faces_around_nodes(np.where(graph.labels == 2)[0], graph.faces)
        outer_adjacent_faces = flatten_list_of_lists(dual.outer_faces())
        invalid_faces = contamination_faces + outer_adjacent_faces

        template = regular_polygon(1.42, 6) / sampling
        rmsd = pairwise_rmsd([template], graph.face_polygons, B_labels=graph.face_labels).ravel()
        defect_faces = rmsd > .07
        defect_faces[invalid_faces] = False

        defects = []
        if np.any(defect_faces):
            dual_defects = dual.subgraph_from_nodes(np.where(defect_faces)[0]).connected_components()

            for dual_defect in dual_defects:
                defect_nodes = select_faces_around_nodes(select_nodes_in_faces(dual_defect.member_nodes, graph.faces),
                                                         graph.faces)
                defect = graph.subgraph_from_faces(defect_nodes).detach()

                dual = defect.dual()
                if len(dual.faces) == 0:
                    continue

                defects.append({})
                defects[-1]['graph'] = defect
                defects[-1]['dual'] = dual
                # TODO : fix connect edges bug, resulting in crash
                defects[-1]['enclosing_path'] = defects[-1]['dual'].outer_face_polygons()[0]
                defects[-1]['contamination'] = len(set(contamination_faces).intersection(defect_nodes)) > 0
                defects[-1]['outside'] = len(set(outer_adjacent_faces).intersection(defect_nodes)) > 0

                defects[-1]['signature'] = 'none'
                if (not defects[-1]['contamination']) & (not defects[-1]['outside']):
                    try:
                        defects[-1]['signature'] = defect_fingerprint(defect, True)
                    except:
                        pass

        return defects


class VisualizationSection(Section):

    def __init__(self, ui):
        super().__init__(ui, 'Visualization')
        self.property_changed_event = Event.Event()

        background_row, self.background_combo_box = combo_box_template(ui, 'Background',
                                                                       ['Image', 'Density', 'Segmentation'])
        self.background = 0
        self.background_combo_box._widget.bind_current_index(Binding.PropertyBinding(self, 'background'))

        point_row, self.point_check_box = check_box_template(ui, 'Points')
        self.overlay_points = True
        self.point_check_box._widget.bind_checked(Binding.PropertyBinding(self, 'overlay_points'))

        point_size_row, self.point_size_line_edit = line_edit_template(ui, 'Point size', default_text=.2)
        self.point_size = .2
        self.point_size_line_edit._widget.bind_text(Binding.PropertyBinding(self, 'point_size'))

        point_color_row, self.point_color_combo_box = combo_box_template(ui, 'Point color', ['Class',
                                                                                             'Solid'])
        self.point_color = 0
        self.point_color_combo_box._widget.bind_current_index(Binding.PropertyBinding(self, 'point_color'))

        graph_row, self.graph_check_box = check_box_template(ui, 'Graph')
        self.overlay_graph = False
        self.graph_check_box._widget.bind_checked(Binding.PropertyBinding(self, 'overlay_graph'))

        outlines_row, self.outlines_check_box = check_box_template(ui, 'Outlines')
        self.overlay_outlines = False
        self.outlines_check_box._widget.bind_checked(Binding.PropertyBinding(self, 'overlay_outlines'))

        tag_row, self.tag_combo_box = combo_box_template(ui, 'Tag',
                                                         ['None', 'Short signature'])
        self.tag = 0
        self.tag_combo_box._widget.bind_current_index(Binding.PropertyBinding(self, 'tag'))

        self.column.add(background_row)
        self.column.add(point_row)
        self.column.add(point_size_row)
        self.column.add(point_color_row)
        self.column.add(graph_row)
        self.column.add(outlines_row)
        self.column.add(tag_row)

    def create_visualization(self, image, sampling, output, graph, defects=None):

        if self.background == 0:
            visualization = array_to_uint8_image(image)

        elif self.background == 1:
            visualization = array_to_uint8_image(output['density'])

        elif self.background == 2:
            visualization = segmentation_to_uint8_image(output['segmentation'])

        else:
            raise RuntimeError()

        point_size = int(float(self.point_size) / sampling)

        if graph is None:
            return visualization

        if self.overlay_graph:
            add_edges(visualization, graph.points, graph.edges, (0, 0, 0))

        if self.overlay_points:
            if self.point_color == 0:
                visualization = add_points(visualization, output['points'], output['labels'], point_size)

            elif self.point_color == 1:
                visualization = add_points(visualization, output['points'], 3, point_size)

            else:
                raise RuntimeError()

        if defects is not None:
            for defect in defects:
                polygon = defect['enclosing_path']
                if self.overlay_outlines:
                    # rectangles = [defect['bbox'] for defect in defects]
                    # visualization = add_rectangles(visualization, rectangles, (255, 0, 0))
                    visualization = add_polygons(visualization, [polygon], (255, 255, 0))

                if self.tag == 1:
                    add_text(visualization, defect['signature'], (np.min(polygon[:, 0]), np.max(polygon[:, 1])),
                             (0, 0, 0))

        return visualization


class LoggerWrapper:

    def __init__(self, logger):
        self.property_changed_event = Event.Event()
        self._logger = logger
        self._int_to_logging_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
        self._logging_level_to_int = {value: key for key, value in self._int_to_logging_level.items()}

    @property
    def logging_level(self):
        return self._logging_level_to_int[self._logger.level]

    @logging_level.setter
    def logging_level(self, value):
        self._logger.setLevel(self._int_to_logging_level[value])


class StructureRecognitionPanelDelegate:

    def __init__(self, api):
        self.api = api
        self.panel_id = "structure-recognition-panel"
        self.panel_name = _("Structure Recognition")
        self.panel_positions = ["left", "right"]
        self.panel_position = "right"
        self.output_data_item = None
        self.source_data_item = None
        self.stop_live_analysis_event = threading.Event()
        self.stop_live_analysis_event.set()
        self._model = None
        self._calibrator = None

    def create_panel_widget(self, ui, document_controller):
        self.ui = ui
        self.document_controller = document_controller
        main_column = ui.create_column_widget()
        scroll_area = ScrollArea(ui._ui)
        scroll_area.content = main_column._widget

        self.deep_learning_section = DeepLearningSection(ui)
        self.scale_section = ScaleDetectionSection(ui)
        self.graph_section = GraphSection(ui)
        self.visualization_section = VisualizationSection(ui)

        log_row, self.log_combo_box = combo_box_template(self.ui, 'Logging level', ['Warning', 'Info', 'Debug'],
                                                         indent=True)
        main_column.add(log_row)

        logger_wrapper = LoggerWrapper(logger)
        logger_wrapper.logging_level = 1
        self.log_combo_box._widget.bind_current_index(Binding.PropertyBinding(logger_wrapper, 'logging_level'))

        run_row, self.run_push_button = push_button_template(ui, 'Start live analysis')

        def start_live_analysis():
            if self.stop_live_analysis_event.is_set():
                self.start_live_analysis()
            else:
                self.stop_live_analysis()

        self.run_push_button.on_clicked = start_live_analysis

        main_column.add(run_row)
        main_column.add(self.deep_learning_section)
        main_column.add(self.scale_section)
        main_column.add(self.graph_section)
        main_column.add(self.visualization_section)

        main_column.add_stretch()

        return scroll_area

    def get_camera(self):
        camera = self.api.get_hardware_source_by_id('superscan', version='1.0')

        if camera is None:
            return self.api.get_hardware_source_by_id('usim_scan_device', '1.0')
        else:
            return camera

    def check_can_analyse_live(self):
        camera = self.get_camera()
        return camera.is_playing

    def start_live_analysis(self):
        if self.get_camera() is None:
            raise RuntimeError('Camera not found')

        if not self.get_camera().is_playing:
            raise RuntimeError('Camera not acquiring')

        self.run_push_button.text = 'Abort live analysis'
        self.stop_live_analysis_event = threading.Event()
        self.process_live()

    def stop_live_analysis(self):
        self.run_push_button.text = 'Start live analysis'
        self.stop_live_analysis_event.set()

    def process_live(self):
        self.output_data_item = self.document_controller.library.create_data_item()
        self.output_data_item.title = 'Visualization of Live Analysis'

        logger.info('Starting live analysis')
        with self.api.library.data_ref_for_data_item(self.output_data_item) as data_ref:

            def thread_this(stop_live_analysis_event, camera, data_ref):
                self.model = load_preset_model('graphene')

                while not stop_live_analysis_event.is_set():
                    if not camera.is_playing:
                        self.stop_live_analysis()

                    source_data = camera.grab_next_to_finish()  # TODO: This starts scanning? Must be a bug.
                    image = source_data[0].data.copy()

                    try:
                        if self.scale_section.calibrate_next:
                            sampling = self.scale_section.calibrate(image, self.model)
                        else:
                            sampling = self.scale_section.sampling
                    except Exception as e:
                        logger.error('Calibration failed: {}'.format(str(e)))
                        sampling = self.scale_section.sampling

                    if sampling is None:
                        logger.error('Pixel size undefined')
                        continue

                    logger.info('Using fov {:.3f} x {:.3f} Angstrom'.format(image.shape[0] * sampling,
                                                                            image.shape[1] * sampling))

                    try:
                        output = self.model(image, sampling)
                    except Exception as e:
                        logger.error('Deep learning model failed: {}'.format(str(e)))
                        continue

                    if output is None:
                        continue

                    try:
                        graph = self.graph_section.build_graph(output['points'], output['labels'], sampling)
                        defects = self.graph_section.analyze_defects(graph, sampling)
                    except Exception as e:
                        logger.error('Graph analysis failed: {}'.format(str(e)))
                        defects = None
                        graph = None

                    try:
                        if output is not None:
                            visualization = self.visualization_section.create_visualization(image, sampling, output,
                                                                                            graph, defects)

                            def update_data_item():
                                data_ref.data = visualization

                            self.api.queue_task(update_data_item)

                    except Exception as e:
                        logger.error('Creating visualization failed: {}'.format(str(e)))

                logger.info('Live analysis ended')

            self.thread = threading.Thread(target=thread_this,
                                           args=(self.stop_live_analysis_event, self.get_camera(), data_ref))
            self.thread.start()


class StructureRecognitionExtension(object):
    extension_id = "nion.swift.extension.structure_recognition"

    def __init__(self, api_broker):
        api = api_broker.get_api(version='~1.0', ui_version='~1.0')
        self.__panel_ref = api.create_panel(StructureRecognitionPanelDelegate(api))

    def close(self):
        self.__panel_ref.close()
        self.__panel_ref = None
