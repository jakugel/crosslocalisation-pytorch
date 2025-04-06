import h5py
import numpy as np


def get_data_labelled(h5filepath, image_size):
    h5_file = h5py.File(h5filepath, "r")

    images = h5_file["images"][0, :]
    images = np.reshape(images, newshape=(-1, image_size, image_size, 1))
    images = np.transpose(images, axes=(0, 3, 2, 1))

    img_labels = h5_file["labels"][0, :]
    img_labels = np.reshape(img_labels, newshape=(-1,))

    return images, img_labels


def get_data_unlabelled(h5filepath, image_size):
    h5_file = h5py.File(h5filepath, "r")

    images = h5_file["images"][1:, :]
    images = np.reshape(images, newshape=(-1, image_size, image_size, 1))
    images = np.transpose(images, axes=(0, 3, 2, 1))

    return images