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
    pickle_file_path = os.path.join(ROOT_DIRECTORY, 'dataset.pickle')

    pickle_file_handle = open(pickle_file_path, 'rb')

    dataset = pickle.load(pickle_file_handle)

    print("Training the Logistic Regression model")

    # Create the Logistic Regression object
    logistic = LogisticRegression()
    # Train the Logistic Regression model with the training set
    logistic.fit(dataset['training_set'], dataset['training_labels'])

    positive_results = 0

    print("Computing accuracy using the test set")

    test_set_index = 0

    for test_set_example in dataset['test_set']:
        prediction = logistic.predict(test_set_example.reshape(1, -1))

        if prediction[0] == dataset['test_labels'][test_set_index]:
            positive_results += 1

        test_set_index += 1

    accuracy = (1.0 * positive_results) / test_set_index

    print("The accuracy of the model on the test set is: ", accuracy)
