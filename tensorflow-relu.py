#!/usr/bin/python
#
# File: tensorflow-relu.py
# Date: 2017-08-18

# ------------------------------------------------------------------------------
# Imports

import tensorflow as tf
import pickle
import os
import numpy as np
import random

# ------------------------------------------------------------------------------
# Constant Parameters

ROOT_DIRECTORY       = '.'
TRAINING_ITERATIONS  = 3500
TRAINING_BATCH_SIZE  = 256
IMAGE_PIXEL_WIDTH    = 28
IMAGE_PIXEL_HEIGHT   = 28
NUMBER_OF_PIXELS     = IMAGE_PIXEL_WIDTH * IMAGE_PIXEL_HEIGHT
HIDDEN_NEURON_COUNT  = 1024
NUMBER_OF_CLASSES    = 10
LEARNING_RATE        = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5]
REGULARIZED_CONSTANT = [0.0, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]

# ------------------------------------------------------------------------------
# Main Logic

if __name__ == "__main__":
    # Read in the training, validation, and test set
    pickle_file_path = os.path.join(ROOT_DIRECTORY, 'dataset.pickle')
    pickle_file_handle = open(pickle_file_path, 'rb')
    dataset = pickle.load(pickle_file_handle)

    learning_rate = tf.placeholder(tf.float32, shape=())
    regularized_constant = tf.placeholder(tf.float32, shape=())

    # Input Layer
    with tf.name_scope("input_layer"):
        inputs = tf.placeholder(tf.float32, [None, NUMBER_OF_PIXELS], name="X")
        weights_1 = tf.Variable(tf.truncated_normal([NUMBER_OF_PIXELS,\
                HIDDEN_NEURON_COUNT], stddev=0.1), name="W")
        bias_1 = tf.Variable(tf.truncated_normal([HIDDEN_NEURON_COUNT],\
                stddev=0.1), name="B")
        outputs_1 = tf.matmul(inputs, weights_1) + bias_1

    # Hidden Layer
    with tf.name_scope("hidden_layer"):
        hidden_layer = tf.nn.relu(outputs_1, name="hidden")

    # Output Layer
    with tf.name_scope("output_layer"):
        weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_NEURON_COUNT,\
                NUMBER_OF_CLASSES], stddev=0.1), name="W")
        bias_2 = tf.Variable(tf.truncated_normal([NUMBER_OF_CLASSES],\
                stddev=0.1), name="B")
        outputs_prediction = tf.nn.softmax(tf.matmul(hidden_layer, weights_2) +\
                bias_2)
        outputs_actual = tf.placeholder(tf.float32, [None, NUMBER_OF_CLASSES],\
                name="labels")

    # Metrics
    with tf.name_scope("metrics"):
        cross_entropy_before_regularization =\
                tf.reduce_mean((-1.0 * tf.reduce_sum(outputs_actual *\
                tf.log(outputs_prediction), reduction_indices=[1])))
        cross_entropy = cross_entropy_before_regularization +\
                (regularized_constant * tf.nn.l2_loss(weights_1)) +\
                (regularized_constant * tf.nn.l2_loss(weights_2))
        cross_prediction = tf.equal(tf.argmax(outputs_prediction, 1),\
                tf.argmax(outputs_actual, 1))
        accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32),\
                name="accuracy")

    # Optimizer
    train_step = tf.train.GradientDescentOptimizer(learning_rate).\
            minimize(cross_entropy)

    # Tensorflow Session
    session = tf.Session()

    # Global Variable Initialization
    init = tf.global_variables_initializer()
    session.run(init)

    # Directory where all parameters/metrics will be written
    writer = tf.summary.FileWriter("./not-mnist-log")
    writer.add_graph(session.graph)

    # Write any additional parameters/metrics
    with tf.name_scope("training"):
        summary_training_accuracy =\
                tf.summary.scalar("accuracy", accuracy)
        summary_training_cross_entropy =\
                tf.summary.scalar("cross entropy", cross_entropy)
        summary_training = tf.summary.merge([summary_training_accuracy,\
                summary_training_cross_entropy])
    with tf.name_scope("validation"):
        summary_validation_accuracy =\
                tf.summary.scalar("accuracy", accuracy)
        summary_validation_cross_entropy =\
                tf.summary.scalar("cross entropy", cross_entropy)
        summary_validation = tf.summary.merge([summary_validation_accuracy,\
                summary_validation_cross_entropy])

    # Initialize where a batch of training dataset will be stored
    batch_x = np.ndarray((TRAINING_BATCH_SIZE, NUMBER_OF_PIXELS))
    batch_y = np.ndarray((TRAINING_BATCH_SIZE, NUMBER_OF_CLASSES))

    training_set_length = len(dataset['training_set'])
    training_set_index = 0
    dataset_index_list = [n for n in range(training_set_length)]
    random.shuffle(dataset_index_list)

    for alpha in LEARNING_RATE:
        for lambda_ in REGULARIZED_CONSTANT:
            writer = tf.summary.FileWriter("./not-mnist-log/" + "alpha" +\
                    str(alpha) + "lambda_" + str(lambda_))

            session.run(init)

            # Training Iterations
            for iteration_index in range(TRAINING_ITERATIONS):
                # Determine if we need to reset the index to the training set
                if training_set_index >= TRAINING_BATCH_SIZE:
                    training_set_index = 0

                # Consilate the examples to be included in the next batch for training
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
                        outputs_actual : batch_y, learning_rate : alpha,\
                        regularized_constant : lambda_})

                if iteration_index % 100 == 0:
                    print("iteration:", iteration_index, "training loss:", l,\
                            "accuracy:", a)
                    summary_training_run = session.run(summary_training, feed_dict=\
                            {inputs : batch_x, outputs_actual : batch_y,\
                            learning_rate : alpha, regularized_constant : lambda_})
                    writer.add_summary(summary_training_run, iteration_index)
                    summary_validation_run = session.run(summary_validation, feed_dict=\
                            {inputs : dataset['validation_set'],\
                            outputs_actual : dataset['validation_labels'],\
                            learning_rate : alpha, regularized_constant : lambda_})
                    writer.add_summary(summary_validation_run, iteration_index)

    print("Training all models complete")
