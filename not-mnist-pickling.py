#!/usr/bin/python
#
# File:         not-mnist-pickling.py
# Date:         2017-08-06

#-------------------------------------------------------------------------------
# Imports
import pickle
import numpy as np
import os
import sys
from scipy import ndimage

ROOT_DIRECTORY                          = '.'
IMAGE_DATASET_DIRECTORY                 = 'notMNIST_large'
PICKLED_DATASET_DIRECTORY               = 'pickled_dataset'
IMAGE_PIXEL_WIDTH                       = 28
IMAGE_PIXEL_HEIGHT                      = 28
IMAGE_PIXEL_DEPTH                       = 255.0
PERCENT_SIZE_TRAINING_SET_PER_LABEL     = 0.15
PERCENT_SIZE_VALIDATION_SET_PER_LABEL   = 0.05
PERCENT_SIZE_TEST_SET_PER_LABEL         = 0.05
NUMBER_OF_CLASSES                       = 10

training_set_size_per_label = {}
validation_set_size_per_label = {}
test_set_size_per_label = {}

training_set_size_total = 0
validation_set_size_total = 0
test_set_size_total = 0

training_set_index = 0
validation_set_index = 0
test_set_index = 0

dataset = {}

if __name__ == "__main__":
    # Check if the image dataset directory exists
    if not os.path.exists(os.path.join(ROOT_DIRECTORY, IMAGE_DATASET_DIRECTORY)):
        print("The image dataset does not exist.  This must be resolved before\
                continuing")
        sys.exit(1)

    # Get a list of the directories that correspond to the different labels
    # and their associated dataset image files
    dataset_directory_list = os.listdir(os.path.join(ROOT_DIRECTORY,\
            IMAGE_DATASET_DIRECTORY))

    print("Computing sizes for the training, validation, and test sets")

    for dataset_directory_index in dataset_directory_list:
        # Compute the path to the dataset for the current label index
        dataset_directory_path = os.path.join(ROOT_DIRECTORY,\
                IMAGE_DATASET_DIRECTORY, dataset_directory_index)
        # Get the total size of the dataset
        dataset_size = len(os.listdir(dataset_directory_path))

        # Compute the number of samples to use for the training set for the
        # currently indexed label
        training_set_size = PERCENT_SIZE_TRAINING_SET_PER_LABEL *\
                dataset_size
        # Track the above computed training set size for the current label
        training_set_size_per_label[dataset_directory_index] =\
                int(training_set_size)
        # Update the total size of the training set accordingly
        training_set_size_total += int(training_set_size)

        # Compute the number of samples to use for the validation set for the
        # currently indexed label
        validation_set_size = PERCENT_SIZE_VALIDATION_SET_PER_LABEL *\
                dataset_size
        # Track the above computed validation set size for the current label
        validation_set_size_per_label[dataset_directory_index] =\
                int(validation_set_size)
        # Update the total size of the validation set accordingly
        validation_set_size_total += int(validation_set_size)

        # Compute the number of samples to use for the test set for the
        # currently indexed label
        test_set_size = PERCENT_SIZE_TEST_SET_PER_LABEL * dataset_size
        # Track the above computed test set size for the current label
        test_set_size_per_label[dataset_directory_index] = int(test_set_size)
        # Update the total size of the test set accordingly
        test_set_size_total += int(test_set_size)

    print("Initializing the dataset structures")
    training_set = np.ndarray((training_set_size_total, IMAGE_PIXEL_WIDTH *\
            IMAGE_PIXEL_HEIGHT))
    training_labels = np.ndarray((training_set_size_total, NUMBER_OF_CLASSES))

    validation_set = np.ndarray((validation_set_size_total, IMAGE_PIXEL_WIDTH *\
            IMAGE_PIXEL_HEIGHT))
    validation_labels = np.ndarray((validation_set_size_total,\
            NUMBER_OF_CLASSES))

    test_set = np.ndarray((test_set_size_total, IMAGE_PIXEL_WIDTH *\
            IMAGE_PIXEL_HEIGHT))
    test_labels = np.ndarray((test_set_size_total, NUMBER_OF_CLASSES))

    print("Curating the datasets")

    # For each label, create the appropriate 'pickled' files from the image
    # dataset
    for dataset_directory_index in dataset_directory_list:
        # For the current label, obtain the path to the label's image dataset
        dataset_directory_path = os.path.join(ROOT_DIRECTORY,\
                IMAGE_DATASET_DIRECTORY, dataset_directory_index)

        # Get a list of the dataset image files of the currently indexed label
        dataset_image_files = os.listdir(dataset_directory_path)

        print("Curating the dataset for label", dataset_directory_path)

        dataset_base_index = 0

        for dataset_image_index in\
                range(training_set_size_per_label[dataset_directory_index]):
            try:
                dataset_image = ndimage.imread(os.path.join(\
                        dataset_directory_path,\
                        dataset_image_files[dataset_base_index +\
                        dataset_image_index])).astype(float)
            except OSError:
                dataset_base_index += 1
                dataset_image = ndimage.imread(os.path.join(\
                        dataset_directory_path,\
                        dataset_image_files[dataset_base_index +\
                        dataset_image_index])).astype(float)

            dataset_image = (dataset_image - IMAGE_PIXEL_DEPTH / 2) /\
                    IMAGE_PIXEL_DEPTH
            training_set[training_set_index, :] = dataset_image.flatten()
            training_labels[training_set_index, :] =\
                    np.eye(1, NUMBER_OF_CLASSES, ord(dataset_directory_index)\
                    - 65).reshape(-1)
            training_set_index += 1

        dataset_base_index +=\
                training_set_size_per_label[dataset_directory_index]

        for dataset_image_index in\
                range(validation_set_size_per_label[dataset_directory_index]):
            try:
                dataset_image = ndimage.imread(os.path.join(\
                        dataset_directory_path,\
                        dataset_image_files[dataset_base_index +\
                        dataset_image_index])).astype(float)
            except OSError:
                dataset_base_index += 1
                dataset_image = ndimage.imread(os.path.join(\
                        dataset_directory_path,\
                        dataset_image_files[dataset_base_index +\
                        dataset_image_index])).astype(float)

            dataset_image = (dataset_image - IMAGE_PIXEL_DEPTH / 2) /\
                    IMAGE_PIXEL_DEPTH
            validation_set[validation_set_index, :] = dataset_image.flatten()
            validation_labels[validation_set_index, :] =\
                    np.eye(1, NUMBER_OF_CLASSES,\
                    ord(dataset_directory_index) - 65).reshape(-1)
            validation_set_index += 1

        dataset_base_index +=\
                validation_set_size_per_label[dataset_directory_index]

        for dataset_image_index in\
                range(test_set_size_per_label[dataset_directory_index]):
            try:
                dataset_image = ndimage.imread(os.path.join(\
                        dataset_directory_path,\
                        dataset_image_files[dataset_base_index +\
                        dataset_image_index])).astype(float)
            except OSError:
                dataset_base_index += 1
                dataset_image = ndimage.imread(os.path.join(\
                        dataset_directory_path,\
                        dataset_image_files[dataset_base_index +\
                        dataset_image_index])).astype(float)

            dataset_image = (dataset_image - IMAGE_PIXEL_DEPTH / 2) /\
                    IMAGE_PIXEL_DEPTH
            test_set[test_set_index, :] = dataset_image.flatten()
            test_labels[test_set_index, :] =\
                    np.eye(1, NUMBER_OF_CLASSES,\
                    ord(dataset_directory_index) - 65).reshape(-1)
            test_set_index += 1

    print("Unifying the training, validation, and test sets")

    dataset['training_set'] = training_set
    dataset['training_labels'] = training_labels
    dataset['validation_set'] = validation_set
    dataset['validation_labels'] = validation_labels
    dataset['test_set'] = test_set
    dataset['test_labels'] = test_labels

    print("Storing the datasets into a 'pickle' file")

    pickle_file_path = os.path.join(ROOT_DIRECTORY, 'dataset.pickle')

    pickle_file_handle = open(pickle_file_path, 'wb')

    pickle.dump(dataset, pickle_file_handle, pickle.HIGHEST_PROTOCOL)

    pickle_file_handle.flush()

    pickle_file_handle.close()
