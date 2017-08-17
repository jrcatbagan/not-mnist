#!/usr/bin/python
import tensorflow as tf
import pickle
import os
import numpy as np
import random

ROOT_DIRECTORY      = '.'
TRAINING_ITERATIONS = 10000
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

    inputs = tf.placeholder(tf.float32, [None, number_of_pixels], name="X")

    with tf.name_scope("input"):
        weights_1 = tf.Variable(tf.truncated_normal([number_of_pixels,\
                HIDDEN_NEURON_COUNT], stddev=0.1), name="W")
        bias_1 = tf.Variable(tf.truncated_normal([HIDDEN_NEURON_COUNT],\
                stddev=0.1), name="B")

    outputs_1 = tf.matmul(inputs, weights_1) + bias_1

    hidden_layer = tf.nn.relu(outputs_1)

    with tf.name_scope("output"):
        weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_NEURON_COUNT,\
                NUMBER_OF_CLASSES], stddev=0.1), name="W")
        bias_2 = tf.Variable(tf.truncated_normal([NUMBER_OF_CLASSES],\
                stddev=0.1), name="B")

    outputs_prediction = tf.nn.softmax(tf.matmul(hidden_layer, weights_2) +\
            bias_2)

    outputs_actual = tf.placeholder(tf.float32, [None, NUMBER_OF_CLASSES],\
            name="labels")

    cross_entropy = tf.reduce_mean((-1.0 * tf.reduce_sum(outputs_actual *\
            tf.log(outputs_prediction), reduction_indices=[1])))

    train_step =\
            tf.train.GradientDescentOptimizer(1E-4).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    session = tf.Session()

    session.run(init)

    writer = tf.summary.FileWriter("./not-mnist-log")
    writer.add_graph(session.graph)

    with tf.name_scope("accuracy"):
        cross_prediction = tf.equal(tf.argmax(outputs_prediction, 1),\
                tf.argmax(outputs_actual, 1))

        accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

    batch_x = np.ndarray((TRAINING_BATCH_SIZE, IMAGE_PIXEL_WIDTH *\
            IMAGE_PIXEL_HEIGHT))
    batch_y = np.ndarray((TRAINING_BATCH_SIZE, NUMBER_OF_CLASSES))

    training_set_length = len(dataset['training_set'])
    training_set_index = 0

    print(training_set_length)

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

        (_, l, a) = session.run([train_step, cross_entropy, accuracy],\
                feed_dict={inputs : batch_x,\
                outputs_actual : batch_y})

        if iteration_index % 100 == 0:
            print("iteration:", iteration_index, "training loss:", l,\
                    "accuracy:", a)


    print(session.run(cross_prediction, feed_dict={inputs : dataset['test_set'],\
            outputs_actual : dataset['test_labels']}))
    print(session.run(accuracy, feed_dict={inputs : dataset['test_set'],\
            outputs_actual : dataset['test_labels']}))
