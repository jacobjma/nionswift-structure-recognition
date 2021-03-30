import bisect

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def greedy_assign(points1, points2, cutoff=np.inf):
    d = cdist(points1, points2)
    d[d > cutoff] = 1e12
    assignment1, assignment2 = linear_sum_assignment(d)
    valid = d[assignment1, assignment2] < cutoff
    return assignment1[valid], assignment2[valid]


def precision_and_recall(points, true_points, mask=None, mask_true=None, cutoff=np.inf, return_indices=False):
    if (len(points) == 0) & (len(true_points) == 0):
        true_positives = []
        false_positives = []
        false_negatives = []
    elif (len(points) == 0):
        true_positives = []
        false_positives = []
        false_negatives = np.arange(len(true_points))
    elif (len(true_points) == 0):
        true_positives = []
        false_positives = np.arange(len(points))
        false_negatives = []
    else:
        assignment, true_assignment = greedy_assign(points, true_points, cutoff)

        true_positives = assignment
        false_positives = np.delete(np.arange(len(points)), assignment, axis=0)
        false_negatives = np.delete(np.arange(len(true_points)), true_assignment, axis=0)

    if mask is not None:
        true_positives = np.intersect1d(true_positives, np.where(mask)[0], assume_unique=True)
        false_positives = np.setdiff1d(false_positives, np.where(mask == 0)[0])

    if mask_true is not None:
        false_negatives = np.setdiff1d(false_negatives, np.where(mask_true == 0)[0], assume_unique=True)

    if len(false_positives) == 0:
        precision = 1.
    else:
        precision = len(true_positives) / (len(true_positives) + len(false_positives))

    if len(false_negatives) == 0:
        recall = 1.
    else:
        recall = len(true_positives) / (len(true_positives) + len(false_negatives))

    if (precision == 0) & (recall == 0):
        f_score = 0.
    else:
        f_score = 2 * (precision * recall) / (precision + recall)

    if return_indices:
        return (precision, recall, f_score), (true_positives, false_positives, false_negatives)
    else:
        return precision, recall, f_score


class HardExampleMiner:

    def __init__(self, max_examples):
        self._max_examples = max_examples
        self._examples = []
        self._num_appended = 0

    def __len__(self):
        return len(self.examples)

    @property
    def examples(self):
        return self._examples

    def reset(self):
        self._examples = []
        self._num_appended = 0

    def append(self, metric, data):
        if len(self) > 0:
            best_hard_metric = self._examples[-1][0]
        else:
            best_hard_metric = np.inf

        if (len(self._examples) < self._max_examples) or (metric < best_hard_metric):

            if len(self._examples) == self._max_examples:
                self._examples.pop(-1)

            bisect.insort(self._examples, (metric, self._num_appended) + (data,))
            self._num_appended += 1