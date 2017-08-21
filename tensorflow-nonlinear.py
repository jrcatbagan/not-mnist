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
        inputs_reshaped = tf.reshape(inputs, [-1, 28, 28, 1])
        weights_input_to_h1 =\
                tf.Variable(tf.truncated_normal([NUMBER_OF_PIXELS,\
                HIDDEN_NEURON_COUNT], stddev=0.1), name="W")
        bias_input_to_h1 = tf.Variable(tf.truncated_normal([HIDDEN_NEURON_COUNT],\
                stddev=0.1), name="B")
        h1_in = tf.matmul(inputs, weights_input_to_h1) + bias_input_to_h1

    # Hidden Layer 1
    with tf.name_scope("hidden_layer_1"):
        h1 = tf.nn.relu(h1_in, name="hidden")
        weights_h1_to_h2 = tf.Variable(tf.truncated_normal([HIDDEN_NEURON_COUNT,\
                HIDDEN_NEURON_COUNT], stddev=0.1), name="W")
        bias_h1_to_h2 = tf.Variable(tf.truncated_normal([HIDDEN_NEURON_COUNT],\
                stddev=0.1), name="B")
        h2_in = tf.matmul(h1, weights_h1_to_h2) + bias_h1_to_h2

    # Hidden Layer 2
    with tf.name_scope("hidden_layer_2"):
        h2 = tf.nn.relu(h2_in)
        weights_h2_to_output = tf.Variable(tf.truncated_normal([HIDDEN_NEURON_COUNT,\
                NUMBER_OF_CLASSES], stddev=0.1))
        bias_h2_to_output = tf.Variable(tf.truncated_normal([NUMBER_OF_CLASSES],\
                stddev=0.1))
        output_in = tf.matmul(h2, weights_h2_to_output) + bias_h2_to_output

    # Output Layer
    with tf.name_scope("output_layer"):
        output = output_in
        output_actual = tf.placeholder(tf.float32, [None, NUMBER_OF_CLASSES],\
                name="labels")

    # Metrics
    with tf.name_scope("metrics"):
        cross_entropy =\
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                labels=output_actual, logits=output))
        cross_entropy_regularized = cross_entropy +\
                (regularized_constant * tf.nn.l2_loss(weights_input_to_h1)) +\
                (regularized_constant * tf.nn.l2_loss(weights_h1_to_h2)) +\
                (regularized_constant * tf.nn.l2_loss(weights_h2_to_output))
        cross_prediction =\
                tf.equal(tf.argmax(output, 1),\
                tf.argmax(output_actual, 1))
        accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32),\
                name="accuracy")

    # Optimizer
    train_step = tf.train.GradientDescentOptimizer(learning_rate).\
            minimize(cross_entropy_regularized)

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
                tf.summary.scalar("cross entropy",\
                cross_entropy_regularized)
        summary_training_image = tf.summary.image("input_image",\
                inputs_reshaped)
        summary_training = tf.summary.merge([summary_training_accuracy,\
                summary_training_cross_entropy, summary_training_image])
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

            print("training with: alpha =", alpha, "-- lambda =", lambda_)
            session.run(init)

            # Training Iterations
            for iteration_index in range(TRAINING_ITERATIONS):
                # Determine if we need to reset the index to the training set
                if training_set_index >= training_set_length:
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

                session.run([train_step],\
                        feed_dict={inputs : batch_x,\
                        output_actual : batch_y, learning_rate : alpha,\
                        regularized_constant : lambda_})

                if iteration_index % 10 == 0:
                    summary_training_run = session.run(summary_training, feed_dict=\
                            {inputs : batch_x, output_actual : batch_y,\
                            learning_rate : alpha, regularized_constant : lambda_})
                    writer.add_summary(summary_training_run, iteration_index)
                    summary_validation_run = session.run(summary_validation, feed_dict=\
                            {inputs : dataset['validation_set'],\
                            output_actual : dataset['validation_labels'],\
                            learning_rate : alpha, regularized_constant : lambda_})
                    writer.add_summary(summary_validation_run, iteration_index)

    print("Training all models complete")
