#!/usr/bin/python
# Incorporate compatibility between python version 2 and version 3
from __future__ import print_function
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
from random import shuffle
import pickle

ROOT_DIRECTORY              = '.'
PICKLED_DATASET_DIRECTORY   = 'pickled_dataset'
IMAGE_PIXEL_WIDTH           = 28
IMAGE_PIXEL_HEIGHT          = 28

if __name__ == '__main__':
    pickled_dataset_directory_path = os.path.join(ROOT_DIRECTORY,\
            PICKLED_DATASET_DIRECTORY)

    pickled_dataset_directory_list = os.listdir(pickled_dataset_directory_path)

    training_set_size = 0
    training_set_index = 0

    training_set = np.ndarray((0))
    training_labels = np.ndarray((0))

    test_set_size = 0
    test_set_index = 0

    test_set = np.ndarray((0))
    test_labels = np.ndarray((0))

    for pickled_dataset_directory_index in pickled_dataset_directory_list:
        pickled_dataset_file_path = os.path.join(pickled_dataset_directory_path,\
                pickled_dataset_directory_index, pickled_dataset_directory_index +\
                '_train.pickle')
        pickled_dataset_file_handle = open(pickled_dataset_file_path, 'rb')

        sub_training_set = pickle.load(pickled_dataset_file_handle)

        pickled_dataset_file_handle.close()

        training_set_size += len(sub_training_set)

        training_set = np.resize(training_set, (training_set_size,\
                IMAGE_PIXEL_WIDTH * IMAGE_PIXEL_HEIGHT))
        training_labels = np.resize(training_labels, (training_set_size))

        for set_index in sub_training_set:
            training_set[training_set_index, :] = set_index
            training_labels[training_set_index] =\
                    int(ord(pickled_dataset_directory_index) - 65)
            training_set_index += 1

    for pickled_dataset_directory_index in pickled_dataset_directory_list:
        pickled_dataset_file_path = os.path.join(pickled_dataset_directory_path,\
                pickled_dataset_directory_index, pickled_dataset_directory_index +\
                '_test.pickle')
        pickled_dataset_file_handle = open(pickled_dataset_file_path, 'rb')

        sub_test_set = pickle.load(pickled_dataset_file_handle)

        pickled_dataset_file_handle.close()

        test_set_size += len(sub_test_set)

        test_set = np.resize(test_set, (test_set_size,\
                IMAGE_PIXEL_WIDTH * IMAGE_PIXEL_HEIGHT))
        test_labels = np.resize(test_labels, (test_set_size))

        for set_index in sub_test_set:
            test_set[test_set_index, :] = set_index
            test_labels[test_set_index] =\
                    int(ord(pickled_dataset_directory_index) - 65)
            test_set_index += 1

    print("Training the Logistic Regression model")

    # Create the Logistic Regression object
    logistic = LogisticRegression()
    # Train the Logistic Regression model with the training set
    logistic.fit(training_set, training_labels)

    positive_results = 0

    print("Computing accuracy using the training set")

    for training_set_index in range(training_set_size):
        prediction = logistic.predict(training_set[training_set_index, :].reshape(1, -1))

        if prediction[0] == training_labels[training_set_index]:
            positive_results += 1

    accuracy = (1.0 * positive_results) / training_set_size

    print("The accuracy of the model on the training set is: ", accuracy)

    positive_results = 0

    print("Computing accuracy using the test set")

    for test_set_index in range(test_set_size):
        prediction = logistic.predict(test_set[test_set_index, :].reshape(1, -1))

        if prediction[0] == test_labels[test_set_index]:
            positive_results += 1

    accuracy = (1.0 * positive_results) / test_set_size

    print("The accuracy of the model on the test set is: ", accuracy)
