#!/usr/bin/python
import tensorflow as tf
import numpy as np
import pickle
import os
import random

ROOT_DIRECTORY      = '.'
TRAINING_ITERATIONS = 5000
TRAINING_BATCH_SIZE = 256
IMAGE_PIXEL_WIDTH   = 28
IMAGE_PIXEL_HEIGHT  = 28
NUMBER_OF_PIXELS    = IMAGE_PIXEL_WIDTH * IMAGE_PIXEL_HEIGHT
HIDDEN_NEURON_COUNT = 1024
NUMBER_OF_CLASSES   = 10


if __name__ == "__main__":
    pickle_file_path = os.path.join(ROOT_DIRECTORY, 'dataset.pickle')
    pickle_file_handle = open(pickle_file_path, 'rb')
    dataset = pickle.load(pickle_file_handle)

    # Input Layer
    with tf.name_scope("input_layer"):
        input_layer = tf.placeholder(tf.float32, [None, NUMBER_OF_PIXELS])
        input_layer_reshaped = tf.reshape(input_layer, [-1, 28, 28, 1])
        weights_input_to_output = tf.Variable(tf.truncated_normal(\
                [NUMBER_OF_PIXELS, NUMBER_OF_CLASSES], stddev=0.1))
        bias_input_to_output = tf.Variable(tf.truncated_normal(\
                [NUMBER_OF_CLASSES], stddev=0.1))

    # Output Layer
    with tf.name_scope("output_layer"):
        output_layer = tf.matmul(input_layer, weights_input_to_output) +\
                bias_input_to_output
        output_actual_layer = tf.placeholder(tf.float32,\
                [None, NUMBER_OF_CLASSES])

        cross_entropy =\
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                labels=output_actual_layer, logits=output_layer))
        cross_prediction =\
                tf.equal(tf.argmax(output_layer, 1),\
                tf.argmax(output_actual_layer, 1))
        accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

    with tf.name_scope("training"):
        summary_training_accuracy = tf.summary.scalar("accuracy", accuracy)
        summary_training_cross_entropy = tf.summary.scalar("cross entropy",\
                cross_entropy)
        summary_training_image = tf.summary.image("input image",\
                input_layer_reshaped)
        summary_training = tf.summary.merge([summary_training_accuracy,\
                summary_training_cross_entropy, summary_training_image])
    with tf.name_scope("validation"):
        summary_validation_accuracy = tf.summary.scalar("accuracy", accuracy)
        summary_validation_cross_entropy = tf.summary.scalar("cross entropy",\
                cross_entropy)
        summary_validation = tf.summary.merge([summary_validation_accuracy,\
                summary_validation_cross_entropy])

    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    session = tf.Session()

    writer = tf.summary.FileWriter("./log-tensorflow-linear", session.graph)

    session.run(init)

    print("training")
    batch_x = np.ndarray((TRAINING_BATCH_SIZE, IMAGE_PIXEL_WIDTH *\
            IMAGE_PIXEL_HEIGHT))
    batch_y = np.ndarray((TRAINING_BATCH_SIZE, NUMBER_OF_CLASSES))

    training_set_length = len(dataset['training_set'])
    training_set_index = 0

    dataset_index_list = [n for n in range(training_set_length)]

    random.shuffle(dataset_index_list)
    random.shuffle(dataset_index_list)
    random.shuffle(dataset_index_list)
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

        session.run([train_step], feed_dict={input_layer : batch_x,\
                output_actual_layer : batch_y})

        if iteration_index % 100 == 0:
            summary_training_run = session.run(summary_training,\
                    feed_dict={input_layer : batch_x,\
                    output_actual_layer : batch_y})
            summary_validation_run = session.run(summary_validation,\
                    feed_dict={input_layer : dataset['validation_set'],\
                    output_actual_layer : dataset['validation_labels']})
            writer.add_summary(summary_training_run, iteration_index)
            writer.add_summary(summary_validation_run, iteration_index)

    print("training complete")
