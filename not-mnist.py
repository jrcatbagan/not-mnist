#!/usr/bin/python
# Incorporate compatibility between python version 2 and version 3
from __future__ import print_function
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import numpy
import os
from PIL import Image
from random import shuffle

DATASET_ROOT = './notMNIST_large'
IMAGE_PIXEL_WIDTH = 28
IMAGE_PIXEL_HEIGHT = 28
NUMBER_OF_SETS_PER_LABEL = 100
PIXEL_DEPTH = 255.0

TRAINING_SET_PERCENT_SIZE_PER_LABEL = 0.1
TEST_SET_PERCENT_SIZE_PER_LABEL = 0.0025

if __name__ == '__main__':
    # Obtain all the dataset directories
    dataset_directory = sorted(os.listdir(DATASET_ROOT))

    # Initialize the variable that will keep track of the total size (i.e.
    # the number of training examples) of the training set
    training_set_size = 0
    # Initialize the variable that will keep track of the total size (i.e.
    # the number of test examples) of the test set
    test_set_size = 0

    print("Computing the number of training sets to obtain for each label")

    label_training_set_lengths = {}

    for dataset_directory_index in dataset_directory:
        directory_path = os.path.join(DATASET_ROOT, dataset_directory_index)
        length = TRAINING_SET_PERCENT_SIZE_PER_LABEL *\
                len(os.listdir(directory_path))
        length = int(length)
        label_training_set_lengths[dataset_directory_index] = length
        training_set_size += length

    print("Computing the number of test sets to obtain for each label")

    label_test_set_lengths = {}
    label_test_set_starting_indices = {}

    for dataset_directory_index in dataset_directory:
        directory_path = os.path.join(DATASET_ROOT, dataset_directory_index)
        length = TEST_SET_PERCENT_SIZE_PER_LABEL *\
                len(os.listdir(directory_path))
        length = int(length)
        label_test_set_lengths[dataset_directory_index] = length
        label_test_set_starting_indices[dataset_directory_index] =\
                label_training_set_lengths[dataset_directory_index]
        test_set_size += length

    print("Initializing the training set")

    # Initialize the training set matrix
    training_set = numpy.ndarray((training_set_size,\
            IMAGE_PIXEL_WIDTH * IMAGE_PIXEL_HEIGHT))
    training_labels = numpy.ndarray((training_set_size), dtype=int)

    training_set_index = 0

    print("Initializing the test set")

    test_set = numpy.ndarray((test_set_size,\
            IMAGE_PIXEL_WIDTH * IMAGE_PIXEL_HEIGHT))
    test_labels = numpy.ndarray((test_set_size), dtype=int)

    test_set_index = 0

    # Index all the dataset directories
    for dataset_directory_index in dataset_directory:
        directory_path = os.path.join(DATASET_ROOT, dataset_directory_index)
        # Obtain the dataset files within the currently indexed dataset
        # directory
        print("Obtaining file index from the directory of label",\
                dataset_directory_index)
        dataset_files = os.listdir(directory_path)

        # Randomize the dataset files
        print("Shuffling file index for label", dataset_directory_index)
        shuffle(dataset_files)

        print("Curating the training set for label", dataset_directory_index)

        for dataset_files_index in\
                range(label_training_set_lengths[dataset_directory_index]):
            image_file = ndimage.imread(os.path.join(directory_path,\
                    dataset_files[dataset_files_index])).astype(float)
            image_file = ((image_file - PIXEL_DEPTH) / 2) / PIXEL_DEPTH

            training_set[training_set_index, :] = image_file.flatten()
            training_labels[training_set_index] = ord(dataset_directory_index) -\
                    65
            training_set_index += 1

        print("Curating the test set for label", dataset_directory_index)

        for dataset_files_index in\
                range(label_training_set_lengths[dataset_directory_index],\
                label_training_set_lengths[dataset_directory_index] +\
                label_test_set_lengths[dataset_directory_index]):
            image_file = ndimage.imread(os.path.join(directory_path,\
                    dataset_files[dataset_files_index])).astype(float)
            image_file = ((image_file - PIXEL_DEPTH) / 2) / PIXEL_DEPTH

            test_set[test_set_index, :] = image_file.flatten()
            test_labels[test_set_index] = ord(dataset_directory_index) - 65
            test_set_index += 1

    print("Training the Logistic Regression model")

    # Create the Logistic Regression object
    logistic = LogisticRegression()
    # Train the Logistic Regression model with the training set
    logistic.fit(training_set, training_labels)

    positive_results = 0

    print("Computing accuracy using the test set")

    for test_set_index in range(test_set_size):
        prediction = logistic.predict(test_set[test_set_index, :].reshape(1, -1))

        if prediction[0] == test_labels[test_set_index]:
            positive_results += 1

    accuracy = (1.0 * positive_results) / test_set_size

    print("The accuracy of the model is: ", accuracy)
