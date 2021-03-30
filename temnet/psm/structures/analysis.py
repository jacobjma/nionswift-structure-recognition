import numpy as np
from scipy.ndimage import center_of_mass
from skimage import measure
from skimage.io import imread
from traitlets import HasTraits, Any, observe, Int, default, Dict, Unicode, Bool
from traittypes import Array

from temnet.psm.coloring import prioritized_greedy_two_coloring
from temnet.psm.dual import faces_to_dual_adjacency
from temnet.psm.graph import stable_delaunay_graph
from temnet.psm.rmsd import batch_rmsd_qcp
from temnet.psm.template import regular_polygon
from temnet.psm.traversal import connected_components
from temnet.psm.utils import is_point_in_polygon, remove_repetition_cyclical, faces_to_double_edge, faces_to_quad_edge
from temnet.psm.utils import order_adjacency_clockwise, flatten_list_of_lists, outer_faces_from_faces, mask_nodes
from temnet.utils import is_position_inside_image
from scipy.ndimage import gaussian_filter


def hbn_reference_counts(reference_path, first_label_to_the_left):
    a = np.array([1, 0])
    b = np.array([-1 / 2., 3 ** 0.5 / 2.])

    new_reference_path = np.dot(reference_path, np.linalg.inv(np.array([a, b])))

    x = np.arange(new_reference_path[:, 0].min() - 1, new_reference_path[:, 0].max() + 2)
    y = np.arange(new_reference_path[:, 1].min() - 1, new_reference_path[:, 1].max() + 2)

    x, y = np.meshgrid(x, y)
    reference_lattice = np.array([x.ravel(), y.ravel()]).T
    reference_lattice = np.dot(reference_lattice, np.array([a, b]))

    basis = np.array([[0, 1 / np.sqrt(3)], [0, -1 / np.sqrt(3)]])

    reference_lattice = reference_lattice[None] + basis[:, None]
    reference_lattice = reference_lattice.reshape((-1, 2))

    reference_labels = np.ones(len(reference_lattice))
    reference_labels[reference_labels.shape[0] // 2:] = 0

    direction = reference_path[1] - reference_path[0]
    direction = direction / np.linalg.norm(direction)
    direction = reference_path[:2].mean(0) + np.array([-direction[1], direction[0]]) * 1 / np.sqrt(3) / 2
    distances = np.linalg.norm(reference_lattice - reference_path[:2].mean(0) + direction, axis=1)

    reference_first_label_to_the_left = reference_labels[distances.argmin()]

    if first_label_to_the_left != reference_first_label_to_the_left:
        reference_labels = reference_labels == 0
    is_in_defect = [is_point_in_polygon(point, reference_path) for point in reference_lattice]

    return (reference_labels[is_in_defect] == 0).sum(), (reference_labels[is_in_defect] == 1).sum()


def step_clockwise_in_hexagonal_lattice(steps):
    a = np.array([1, 0])
    b = np.array([-1 / 2., 3 ** 0.5 / 2.])
    vectors = np.array([a, -b, -a - b, - a, b, a + b])
    k = 0
    reference_path = [vectors[k]]
    for i in range(len(steps)):
        j = (k + steps[i] + 3) % len(vectors)
        reference_path.append(reference_path[-1] + vectors[j])
        k = j

    return np.array(reference_path)


def assign_sublattice_hexagonal(adjacency, points, lattice_constant):
    template = regular_polygon(3, lattice_constant, append_center=True)
    segments = [[node] + adjacent for node, adjacent in adjacency.items()]
    three_connected = np.array([len(segment) == 4 for segment in segments])

    rmsd = np.zeros(len(segments))
    rmsd[three_connected == 0] = np.inf
    segments = points[np.array([segment for i, segment in enumerate(segments) if three_connected[i]])]
    segments -= segments[:, 0, None]

    rmsd[three_connected] = batch_rmsd_qcp(template[None], segments)
    labels = prioritized_greedy_two_coloring(adjacency, rmsd)
    return labels


def assign_sublattice_hbn(graph, image, sampling):
    adjacency = graph.adjacency
    points = graph.points

    labels = assign_sublattice_hexagonal(adjacency, points, 2.504 / sampling)
    inside = is_position_inside_image(points, image.shape, -5)
    points_in_image = np.round(points[inside]).astype(np.int)

    point_intensity = gaussian_filter(image, 8)[points_in_image[:, 1], points_in_image[:, 0]]
    relative_intensity = point_intensity[labels[inside] == 1].mean() / point_intensity[labels[inside] == 0].mean()
    if relative_intensity < 1:
        labels = (labels == 0).astype(np.int)

    return labels


def connected_equalsized_faces(faces, face_centers, size, negate=False, discard_outer=True, return_boundaries=False):
    if negate:
        equal_adjacent_nodes = np.array([len(face) for face in faces]) != size
    else:
        equal_adjacent_nodes = np.array([len(face) for face in faces]) == size

    if discard_outer:
        outer = flatten_list_of_lists(outer_faces_from_faces(faces))
        outer_adjacent = [True if not set(face).intersection(outer) else False for face in faces]
        equal_adjacent_nodes *= outer_adjacent

    dual_adjacency = order_adjacency_clockwise(face_centers, faces_to_dual_adjacency(faces))
    components = connected_components(mask_nodes(dual_adjacency, equal_adjacent_nodes))

    if return_boundaries:
        return [outer_faces_from_faces([faces[x] for x in component])[0] for component in components]
    else:
        return components


def hbn_defect_metrics(points, labels, faces, defect_boundaries):
    face_centers = np.array([points[face].mean(0) for face in faces])
    double_edge = faces_to_double_edge(faces)
    reverse_quad_edge = {tuple(value): key for key, value in faces_to_quad_edge(faces).items()}

    metrics = []
    for boundary in defect_boundaries:
        path = []
        closed = True
        for i, j in zip(boundary, np.roll(boundary, -1)):
            try:
                path.append(double_edge[(i, j)])
            except KeyError:
                closed = False

        path, repetitions = remove_repetition_cyclical(path)

        try:
            first_label_to_the_left = reverse_quad_edge[(path[0], path[1])][0]
        except KeyError:
            closed = False

        if closed:
            reference_path = step_clockwise_in_hexagonal_lattice(np.array(repetitions) + 1)
            reference_count = hbn_reference_counts(reference_path, labels[first_label_to_the_left])

            is_in_defect = [is_point_in_polygon(point, face_centers[path]) for point in points]

            count = [(labels[is_in_defect] == 0).sum(), (labels[is_in_defect] == 1).sum()]

            missing = (reference_count[0] - count[0], reference_count[1] - count[1])
        else:
            missing = (0, 0)

        center = face_centers[path[:-1]].mean(0)

        metrics.append({'boundary': path,
                        'center': center,
                        'num_missing': missing,
                        'B_vacancy': missing == (1, 0),
                        'N_vacancy': missing == (0, 1)})

    return metrics


class TimeSeriesData:

    def process_series(self, images, temnet):
        for image in images:
            self.get_defects(i)


class hBNTimeSeriesData:

    def __init__(self):
        pass

    def write(self, fname):
        import pickle
        pickle.dump(self, open(fname, "wb"))

    @classmethod
    def read(self, fname):
        import pickle
        return pickle.load(open(fname, "rb"))

    def process_series(self, images, temnet):
        for i in range(len(self._images)):
            self.get_defects(i)

    def get_frame(self, i):
        return self._images[i]

    def get_sampling(self, i):
        return 0.0339

    def _calculated_data(self, i, key, calculator):
        try:
            return self._frame_data[i][key]
        except:
            self._frame_data[i][key] = calculator(i)
            return self._frame_data[i][key]

    def process_frame(self, i, image, temnet):
        points, sums, segments = self._temnet(self.get_frame(i), self.get_sampling(i))

        return points

    def set_points(self, i, points):
        self._frame_data[i].pop('labels', None)
        self._frame_data[i].pop('graph', None)
        self._frame_data[i]['points'] = points

    def get_points(self, i):
        return self._calculated_data(i, 'points', self.calculate_points)

    def get_segmentation(self, i):
        self.get_points(i)
        return self._frame_data[i]['segmentation']

    def calculate_graph(self, i):
        graph = stable_delaunay_graph(self.get_points(i), alpha=2)
        return graph

    def get_graph(self, i):
        return self._calculated_data(i, 'graph', self.calculate_graph)

    def calculate_labels(self, i):
        image = self.get_frame(i)
        points = self.get_points(i)
        adjacency = self.get_graph(i).adjacency

        labels = assign_sublattice_hexagonal(adjacency, points, 2.504 / self.get_sampling(i))
        inside = is_position_inside_image(points, image.shape, -5)
        points_in_image = np.round(points[inside]).astype(np.int)
        from scipy.ndimage import gaussian_filter
        point_intensity = gaussian_filter(image, 8)[points_in_image[:, 1], points_in_image[:, 0]]

        relative_intensity = point_intensity[labels[inside] == 1].mean() / point_intensity[labels[inside] == 0].mean()

        if relative_intensity < 1:
            labels = (labels == 0).astype(np.int)

        return labels

    def get_labels(self, i):
        return self._calculated_data(i, 'labels', self.calculate_labels)

    def calculate_defects(self, i):
        segmentation = self.get_segmentation(i)

        defects = {}
        defect_num = 0
        for label in range(1, labels.max() + 1):

            defect_segment = labels == label

            area = (defect_segment.sum() / area_per_vacancy).item()

            if area < .5:
                continue

            contour = measure.find_contours(defect_segment, .5)[0][:, ::-1]
            contour = self._temnet.unpad_points(contour, self._images.shape, self.get_sampling(i))

            if (np.all(contour[:, 0] < 0) or np.all(contour[:, 1] < 0) or
                    np.all(contour[:, 0] > self._images.shape[-2]) or
                    np.all(contour[:, 1] > self._images.shape[-1])):
                continue

            defects[defect_num] = {}
            defects[defect_num]['area'] = round(area, 3)

            defects[defect_num]['contour'] = contour

            if (np.any(contour < 0) or
                    np.any(contour[:, 0] > self._images.shape[-2]) or
                    np.any(contour[:, 1] > self._images.shape[-1])):

                defects[defect_num]['boundary'] = True
            else:
                defects[defect_num]['boundary'] = False

            center = center_of_mass(defect_segment)
            center = self._temnet.unpad_points(np.array([[center[1], center[0]]]), self._images.shape,
                                               self.get_sampling(i))

            defects[defect_num]['center'] = (round(center[0][0], 3), round(center[0][1], 3))
            defects[defect_num]['type'] = 'unclassified'

            # if defects[label]['boundary'] == False:
            points = np.vstack((center, self.get_points(i)))
            graph = stable_delaunay_graph(points, 2)
            adjacent_labels = [self.get_labels(i)[j - 1] for j in graph.adjacency[0]]

            if adjacent_labels == [1, 1, 1]:
                defects[defect_num]['type'] = 'B_vacancy'
            elif adjacent_labels == [0, 0, 0]:
                defects[defect_num]['type'] = 'N_vacancy'

            defect_num += 1
            # if (len(adjacent_labels) > 0) & (defects[label]['type'] == 'unknown'):
            #     adjacent_label = most_common(adjacent_labels)
            #
            #     if adjacent_label == 0:
            #         defects[label]['type'] = 'possible_B_vacancy'
            #     elif adjacent_label == 1:
            #         defects[label]['type'] = 'possible_N_vacancy'

        return defects  # [defect]
        # faces = self.get_graph(i).faces
        # face_centers = self.get_graph(i).face_centers
        # points = self.get_points(i)
        # labels = self.get_labels(i)
        # dual_faces = self.get_graph(i).dual.faces
        # # print(dual_faces[40])
        #
        # defect_boundaries = connected_equalsized_faces(faces, face_centers, 6, negate=True, discard_outer=False,
        #                                                return_boundaries=True)
        # # print(connected)
        # # expanded = [flatten_list_of_lists([dual_adjacency[face] for face in component]) for component in connected]
        #
        # # print(expanded)
        # # defect_boundaries = [outer_faces_from_faces(faces[face_idx])[0] for face_idx in expanded]
        # defect_boundaries = order_faces_clockwise(defect_boundaries, points, True)
        #
        # return hbn_defect_metrics(points, labels, faces, defect_boundaries)

    def get_defects(self, i):
        return self._calculated_data(i, 'defects', self.calculate_defects)

    def get_frame_summaries(self):
        frame_summaries = {}
        frame_summaries['defect'] = []

        for i in range(len(self._images)):
            defects = self.get_defects(i)
            area = np.sum([defect['area'] for defect in defects.values()])
            frame_summaries['defect'].append(area)

        return frame_summaries

    def get_summary(self):
        summary = {}

        summary['defect_created'] = False
        summary['first_defect_frame'] = None
        summary['num_frames'] = len(self._images)
        for i in range(len(self._images)):
            defects = self.get_defects(i)

            if len(defects) > 0:
                summary['defect_created'] = True

                if summary['first_defect_frame'] is None:
                    summary['first_defect_frame'] = i

                    if len(defects) > 1:
                        summary['first_defect_type'] = 'multiple'
                        summary['first_defect_center'] = None
                    else:
                        summary['first_defect_type'] = defects[0]['type']
                        summary['first_defect_center'] = defects[0]['center']

        summary['skipped'] = False
        return summary


class hBNTimeSeriesTraits(HasTraits):
    current_fname = Unicode()
    current_index = Int(0)
    recalculate = Bool()

    timeseries_data = Any()
    temnet = Any()

    points = Array(allow_none=True)
    image = Array()
    density = Array()
    labels = Array()
    graph = Any()
    defect_lines = Array()
    timeline_data = Dict()
    defect_data = Dict()
    timeseries_summary = Dict()

    @default('image')
    def _default_image(self):
        return np.zeros((0, 0))

    @default('points')
    def _default_points(self):
        return np.zeros((0, 2))

    @observe('current_index')
    def _observe_current_index(self, change):
        self.points = self.time_series_data.get_points(self.current_index)
        self.image = self.time_series_data.get_frame(self.current_index)
        frame_summaries = self.time_series_data.get_frame_summaries()

        self.timeline_data = frame_summaries
        defect_data = self.time_series_data.get_defects(self.current_index)

        keys = ['area', 'boundary', 'center', 'type']
        defect_data = {key: {inner_key: value[inner_key] for inner_key in keys} for key, value in defect_data.items()}

        self.defect_data = defect_data

        summary = self.time_series_data.get_summary()
        self.timeseries_summary = summary

        density = self.time_series_data.get_density(self.current_index)
        self.density = density

    @observe('current_fname')
    def _observe_current_fname(self, change):
        images = imread(self.current_fname)
        if images.shape[-1] == 3:
            images = images[None, :, :, 0]

        import os
        if not self.recalculate:
            try:
                time_series_data = hBNTimeSeriesData.read(os.path.splitext(self.current_fname)[0] + '.p')
                time_series_data._images = images
                time_series_data._temnet = self.temnet
                self.time_series_data = time_series_data
                # print('read from disk')
            except:
                self.time_series_data = hBNTimeSeriesData(images=images, temnet=self.temnet)
        else:
            self.time_series_data = hBNTimeSeriesData(images=images, temnet=self.temnet)

        try:
            self.time_series_data.process_series()
            self.current_index = 0
            self._observe_current_index(None)
        except:
            self.timeseries_summary = {'skipped': True}

        self.time_series_data.write(os.path.splitext(self.current_fname)[0] + '.p')

    #     @observe('edit')
    #     def _observe_edit(self):
    #         if self.edit:
    #             directional_link((point_artist, 'edited_points'), (time_series_traits, 'edited_points'))

    # @observe('edited_points')
    #     def _observe_edited_points(self, change):

    #         self.time_series_data.set_points(self.current_index, self.edited_points)

    #         self._observe_points(None)

    @observe('points')
    def _observe_points(self, change):
        self.labels = self.time_series_data.get_labels(self.current_index)
        self.graph = self.time_series_data.get_graph(self.current_index)
        defects = self.time_series_data.get_defects(self.current_index)
        self.defect_lines = [defect['contour'] for defect in defects.values()]
        # [face_centers[defect['boundary'] + [defect['boundary'][0]]] for defect in defects]
