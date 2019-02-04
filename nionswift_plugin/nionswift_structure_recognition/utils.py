import numpy as np


def standardize_image(image):
    image -= np.min(image)
    image /= np.std(image)
    return image


def ensemble_expand(image):
    ensemble = np.zeros((8,) + image.shape)

    ensemble[0] = image
    ensemble[1] = np.fliplr(image)
    ensemble[2] = np.flipud(image)
    ensemble[3] = np.rot90(image)
    ensemble[4] = np.fliplr(np.flipud(image))
    ensemble[5] = np.fliplr(np.rot90(image))
    ensemble[6] = np.fliplr(np.flipud(np.rot90(image)))
    ensemble[7] = np.flipud(np.rot90(image))

    return ensemble


def ensemble_reduce(ensemble):
    ensemble[1] = np.fliplr(ensemble[1])
    ensemble[2] = np.flipud(ensemble[2])
    ensemble[3] = np.rot90(ensemble[3], k=3)
    ensemble[4] = np.flipud(np.fliplr(ensemble[4]))
    ensemble[5] = np.rot90(np.fliplr(ensemble[5]), k=3)
    ensemble[6] = np.rot90(np.flipud(np.fliplr(ensemble[6])), k=3)
    ensemble[7] = np.rot90(np.flipud(ensemble[7]), k=3)

    return np.sum(ensemble, axis=0) / 8.
