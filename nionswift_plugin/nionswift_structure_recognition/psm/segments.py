import numpy as np
import numba as nb


def list_of_indices_to_run_length(segment_list):
    size = 0
    for segment in segment_list:
        size += len(segment)

    indices = np.zeros(size, dtype=np.int32)
    fronts = np.zeros(len(segment_list), dtype=np.int32)

    j = 0
    for i, segment in enumerate(segment_list):
        k = len(segment)
        indices[j:j + k] = segment
        fronts[i] = j
        j = j + k
    return indices, fronts


@nb.njit
def segment_centers(points, indices, partitions):
    centers = np.zeros((len(partitions) - 1, 2))
    for i in range(len(partitions) - 1):
        centers[i] = np.sum(points[indices[partitions[i]:partitions[i + 1]]], axis=0)
        centers[i] = centers[i] / (partitions[i + 1] - partitions[i])
    return centers


class RunLengthSegments(object):

    def __init__(self, indices=None, partitions=None):
        self._indices = indices
        self._partitions = partitions

    def __len__(self):
        return len(self._partitions) - 1

    @property
    def indices(self):
        return self._indices

    @property
    def partitions(self):
        return self._partitions

    @property
    def fronts(self):
        return self._partitions[:-1]

    def sizes(self):
        return np.diff(self.partitions)

    def __getitem__(self, i):

        if isinstance(i, slice):

            start = i.start
            stop = i.stop
            if start is None:
                start = 0

            if stop is None:
                stop = len(self)

            if stop < 0:
                stop = (stop % len(self)) + 1

            indices = self.indices[self.partitions[start]:self.partitions[stop]]
            partitions = self.partitions[start:stop] - self.partitions[start]
            return self.__class__(indices, partitions)

        else:
            if i < 0:
                i = i % len(self)
            return self.indices[self._partitions[i]:self._partitions[i + 1]]
        # else:
        #     raise RuntimeError()
