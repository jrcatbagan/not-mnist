#!/usr/bin/python
import tensorflow as tf
import numpy as np
import pickle
import os

ROOT_DIRECTORY     = '.'
TRAINING_ITERATIONS = 50000
TRAINING_BATCH_SIZE = 256
IMAGE_PIXEL_WIDTH   = 28
IMAGE_PIXEL_HEIGHT  = 28
HIDDEN_NEURON_COUNT = 1024
NUMBER_OF_CLASSES   = 10


if __name__ == "__main__":
    pickle_file_path = os.path.join(ROOT_DIRECTORY, 'dataset.pickle')

    pickle_file_handle = open(pickle_file_path, 'rb')

    dataset = pickle.load(pickle_file_handle)

    number_of_pixels = IMAGE_PIXEL_WIDTH * IMAGE_PIXEL_HEIGHT

    # Initialize the structure used for the inputs
    inputs = tf.placeholder(tf.float32, [None, number_of_pixels])
    # Initialize the structure used for the weight parameters
    W = tf.Variable(tf.zeros([number_of_pixels, 10]))
    # Initialize the structure used for the bias parameters
    b = tf.Variable(tf.zeros([10]))

    outputs_prediction = tf.nn.softmax(tf.matmul(inputs, W) + b)

    outputs_actual = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean((-1.0 * tf.reduce_sum(outputs_actual * tf.log(outputs_prediction),\
            reduction_indices=[1])))

    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    session = tf.Session()

    session.run(init)

    print("training")
    batch_x = np.ndarray((TRAINING_BATCH_SIZE, IMAGE_PIXEL_WIDTH *\
            IMAGE_PIXEL_HEIGHT))
    batch_y = np.ndarray((TRAINING_BATCH_SIZE, NUMBER_OF_CLASSES))

    training_set_length = len(dataset['training_set'])
    training_set_index = 0

    for iteration_index in range(TRAINING_ITERATIONS):
        if (training_set_index + TRAINING_BATCH_SIZE) < training_set_length:
            batch_x[:,:] =\
                    dataset['training_set'][training_set_index:\
                    training_set_index + TRAINING_BATCH_SIZE,:]
            batch_y[:,:] =\
                    dataset['training_labels'][training_set_index:\
                    training_set_index + TRAINING_BATCH_SIZE,:]
            training_set_index += TRAINING_BATCH_SIZE
        else:
            n = training_set_length - training_set_index
            batch_x[:n, :] =\
                    dataset['training_set'][training_set_index:\
                    training_set_length, :]
            batch_y[:n, :] =\
                    dataset['training_labels'][training_set_index:\
                    training_set_length, :]
            training_set_index = 0

            batch_x[n:, :] =\
                    dataset['training_set'][training_set_index:\
                    TRAINING_BATCH_SIZE - n, :]
            batch_y[n:, :] =\
                    dataset['training_labels'][training_set_index:\
                    TRAINING_BATCH_SIZE - n, :]
            training_set_index += TRAINING_BATCH_SIZE - n

        (_, l) = session.run([train_step, cross_entropy],\
                feed_dict={inputs : batch_x,\
                outputs_actual : batch_y})

        print("iteration:", iteration_index, " - ", "loss:", l)

    cross_prediction = tf.equal(tf.argmax(outputs_prediction, 1),\
            tf.argmax(outputs_actual, 1))

    accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

    print(session.run(accuracy, feed_dict={inputs : dataset['test_set'],\
            outputs_actual : dataset['test_labels']}))
