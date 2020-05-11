import numpy as np


def label_to_index_generator(labels, first_label=0):
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(first_label, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield i, np.sort(indices[l:h])


def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def labels_to_lists(labels, labelled):
    labelled = np.array(labelled)
    lists = []
    for i, indices in label_to_index_generator(labels):
        lists.append(labelled[indices])
    return lists
