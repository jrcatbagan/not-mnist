#!/usr/bin/python
#
# File:         not-mnist-pickling.py
# Date:         2017-08-06

#-------------------------------------------------------------------------------
# Imports
from __future__ import print_function
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
PERCENT_SIZE_VALIDATION_SET_PER_LABEL   = 0.0
PERCENT_SIZE_TEST_SET_PER_LABEL         = 0.05

if __name__ == "__main__":
    image_dataset_directory_status = os.path.exists(os.path.join(ROOT_DIRECTORY,\
            IMAGE_DATASET_DIRECTORY))

    # Check if the image dataset directory exists
    if not os.path.exists(os.path.join(ROOT_DIRECTORY, IMAGE_DATASET_DIRECTORY)):
        print("The image dataset does not exist.  This must be resolved before\
                continuing")
        sys.exit(1)

    # Create the directory where the 'pickled' files will be store, if it does
    # not exist yet that is
    if not os.path.exists(os.path.join(ROOT_DIRECTORY,\
            PICKLED_DATASET_DIRECTORY)):
        os.makedirs(os.path.join(ROOT_DIRECTORY,PICKLED_DATASET_DIRECTORY))

    # Get a list of the directories that correspond to the different labels
    # and their associated dataset image files
    image_dataset_directory_list = os.listdir(os.path.join(ROOT_DIRECTORY,\
            IMAGE_DATASET_DIRECTORY))

    print("Pickling the datasets")
    # For each label, create the appropriate 'pickled' files from the image
    # dataset
    for directory_index in image_dataset_directory_list:
        # For the current label, obtain the path to the label's image dataset
        image_dataset_directory_path = os.path.join(ROOT_DIRECTORY,\
                IMAGE_DATASET_DIRECTORY, directory_index)
        # Compute the number of images to extract based on a percentage of the
        # total number of images for the current label
        image_dataset_training_length = PERCENT_SIZE_TRAINING_SET_PER_LABEL *\
                len(os.listdir(image_dataset_directory_path))
        image_dataset_training_length = int(image_dataset_training_length)
        # Get a list of all the image dataset files for the current label
        image_dataset_files = os.listdir(image_dataset_directory_path)

        # Initialize the structure for the images of the current label to be
        # used as the training set.  The first dimension are the indices to the
        # images and the second dimension are the image pixel matrix flattened.
        training_set = np.ndarray((image_dataset_training_length,\
                IMAGE_PIXEL_WIDTH * IMAGE_PIXEL_HEIGHT))

        image_dataset_test_length = PERCENT_SIZE_TEST_SET_PER_LABEL *\
                len(os.listdir(image_dataset_directory_path))
        image_dataset_test_length = int(image_dataset_test_length)
        test_set = np.ndarray((image_dataset_test_length,\
                IMAGE_PIXEL_WIDTH * IMAGE_PIXEL_HEIGHT))

        # For each image dataset file of the current label, store it into the
        # previously created structures appropriately
        for image_dataset_files_index in range(image_dataset_training_length):
            # Get the path to the currently indexed image dataset file of the
            # currently indexed label
            image_dataset_file = ndimage.imread(os.path.join(\
                    image_dataset_directory_path,\
                    image_dataset_files[image_dataset_files_index])).astype(float)
            # Read in the currently indexed image dataset file and normalize it
            image_dataset_file = ((image_dataset_file - IMAGE_PIXEL_DEPTH) / 2) /\
                    IMAGE_PIXEL_DEPTH

            # Store the normalized image dataset file
            training_set[image_dataset_files_index,:] =\
                    image_dataset_file.flatten()

        for image_dataset_files_index in range(image_dataset_training_length,\
                image_dataset_training_length + image_dataset_test_length):
            try:
                image_dataset_file = ndimage.imread(os.path.join(\
                        image_dataset_directory_path,\
                        image_dataset_files[image_dataset_files_index])).astype(float)
                image_dataset_file = ((image_dataset_file - IMAGE_PIXEL_DEPTH) / 2) /\
                        IMAGE_PIXEL_DEPTH
                test_set[image_dataset_files_index - image_dataset_training_length, :] =\
                        image_dataset_file.flatten()
            except OSError:
                pass


        pickled_dataset_directory_path = os.path.join(ROOT_DIRECTORY,\
                PICKLED_DATASET_DIRECTORY, directory_index)
        os.makedirs(pickled_dataset_directory_path, exist_ok=True)
        # Get the path to the 'pickled' file where the training set of the
        # current label will be stored
        pickled_dataset_file_path = os.path.join(pickled_dataset_directory_path,\
                directory_index + '_train.pickle')

        pickled_dataset_file_handle = open(pickled_dataset_file_path, 'wb')

        print("Storing the training set for the label",\
                directory_index)
        pickle.dump(training_set, pickled_dataset_file_handle,\
                pickle.HIGHEST_PROTOCOL)

        pickled_dataset_file_handle.flush()

        pickled_dataset_file_handle.close()

        pickled_dataset_file_path = os.path.join(pickled_dataset_directory_path,\
                directory_index + '_test.pickle')

        pickled_dataset_file_handle = open(pickled_dataset_file_path, 'wb')

        print("Storing the test set for the label",\
                directory_index)
        pickle.dump(test_set, pickled_dataset_file_handle,\
                pickle.HIGHEST_PROTOCOL)

        pickled_dataset_file_handle.flush()

        pickled_dataset_file_handle.close()
