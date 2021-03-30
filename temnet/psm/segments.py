import numpy as np
import scipy.spatial as spatial


def validate_points(points):
    try:
        return points.points
    except:
        return points


def stack(tup):
    try:
        points = np.vstack([points.points for points in tup])
        labels = np.concatenate([points.labels for points in tup])
        return LabelledPoints(points, labels)
    except:
        return np.vstack(tup)


class LabelledPoints:

    def __init__(self, points, labels):
        self._points = points
        self._labels = labels

    @property
    def points(self):
        return self._points

    @property
    def labels(self):
        return self._labels

    def __sub__(self, other):
        return self.__class__(self.points - other.points, self.labels)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            labels = self.labels[item[0]]
        else:

            labels = self.labels[item]

        return self.__class__(self._points[item], labels)


def radius_nearest_neighbor_segments(points, centers, radius):
    point_tree = spatial.cKDTree(validate_points(points))
    segments = point_tree.query_ball_point(validate_points(centers), radius)

    segments = [np.array(segment)[np.argsort(((validate_points(points)[segment] - center) ** 2).sum(-1))]
                for segment, center in zip(segments, validate_points(centers))]

    return SegmentedPoints(points, segments)


class SegmentedPoints:

    def __init__(self, points, segments=None):
        if isinstance(points, list):
            sizes = [len(segment) for segment in points]
            cum_sizes = np.cumsum([0] + sizes)
            segments = [np.arange(cum_size, cum_size + size) for size, cum_size in zip(sizes, cum_sizes)]
            points = np.concatenate(points)

        self._points = points
        self._segments = np.array(segments, dtype='object')

    @property
    def points(self):
        return self._points

    @property
    def segments(self):
        return self._segments

    def __getitem__(self, item):
        points = self.points[np.hstack(self.segments[item]).astype(np.int)]
        return points

    def __len__(self):
        return len(self.segments)

    @property
    def sizes(self):
        return [len(segment) for segment in self.segments]

    def extract(self, center=None):
        if center:
            return [self.points[segment]  - self.points[segment[0]] for segment in self.segments]
        else:
            return [self.points[segment] for segment in self.segments]
