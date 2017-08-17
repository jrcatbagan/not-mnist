#!/usr/bin/python
import tensorflow as tf
import numpy as np
import pickle
import os
import random

ROOT_DIRECTORY     = '.'
TRAINING_ITERATIONS = 2000000
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
    W = tf.Variable(tf.truncated_normal([number_of_pixels, 10], stddev=0.1))
    # Initialize the structure used for the bias parameters
    b = tf.Variable(tf.zeros([10]))

    outputs_prediction = tf.nn.softmax(tf.matmul(inputs, W) + b)

    outputs_actual = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean((-1.0 * tf.reduce_sum(outputs_actual * tf.log(outputs_prediction),\
            reduction_indices=[1])))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    session = tf.Session()

    writer = tf.summary.FileWriter("./log-tensorflow-linear", session.graph)

    session.run(init)

    cross_entropy_summary = tf.summary.scalar("cross entropy",\
            cross_entropy)

    cross_prediction = tf.equal(tf.argmax(outputs_prediction, 1),\
            tf.argmax(outputs_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

    accuracy_summary = tf.summary.scalar("accuracy", accuracy)

    merged_summaries = tf.summary.merge_all()

    print("training")
    batch_x = np.ndarray((TRAINING_BATCH_SIZE, IMAGE_PIXEL_WIDTH *\
            IMAGE_PIXEL_HEIGHT))
    batch_y = np.ndarray((TRAINING_BATCH_SIZE, NUMBER_OF_CLASSES))

    training_set_length = len(dataset['training_set'])
    training_set_index = 0

    dataset_index_list = [n for n in range(training_set_length)]

    random.shuffle(dataset_index_list)
    for iteration_index in range(TRAINING_ITERATIONS):
        if training_set_index >= TRAINING_BATCH_SIZE:
            training_set_index = 0

        for batch_index in range(TRAINING_BATCH_SIZE):
            batch_x[batch_index, :] =\
                    dataset['training_set'][dataset_index_list[training_set_index],\
                    :]
            batch_y[batch_index, :] =\
                    dataset['training_labels'][dataset_index_list[training_set_index],\
                    :]
            training_set_index += 1

        (_, l) = session.run([train_step, cross_entropy],\
                feed_dict={inputs : batch_x,\
                outputs_actual : batch_y})

        if iteration_index % 100 == 0:
            print("iteration:", iteration_index, " - ", "loss:", l)
            s = session.run(merged_summaries, feed_dict=\
                    {inputs : batch_x, outputs_actual : batch_y})
            writer.add_summary(s, iteration_index)

    cross_prediction = tf.equal(tf.argmax(outputs_prediction, 1),\
            tf.argmax(outputs_actual, 1))

    accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

    print("training accuracy:", session.run(accuracy,\
            feed_dict={inputs : dataset['training_set'],\
            outputs_actual : dataset['training_labels']}))
    print("validation accuracy:", session.run(accuracy,\
            feed_dict={inputs : dataset['validation_set'],\
            outputs_actual : dataset['validation_labels']}))
    print("test accuracy:", session.run(accuracy,\
            feed_dict={inputs : dataset['test_set'],\
            outputs_actual : dataset['test_labels']}))
