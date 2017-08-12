#!/usr/bin/python
import tensorflow as tf
import numpy as np
import pickle
import os

ROOT_DIRECTORY     = '.'
IMAGE_PIXEL_WIDTH  = 28
IMAGE_PIXEL_HEIGHT = 28
ITERATIONS         = 1000
BATCH_SIZE         = 100

if __name__ == "__main__":
    pickle_file_path = os.path.join(ROOT_DIRECTORY, 'dataset.pickle')

    pickle_file_handle = open(pickle_file_path, 'rb')

    dataset = pickle.load(pickle_file_handle)

    number_of_pixels = IMAGE_PIXEL_WIDTH * IMAGE_PIXEL_HEIGHT

    # Initialize the structure used for the inputs
    X = tf.placeholder(tf.float32, [None, number_of_pixels])
    # Initialize the structure used for the weight parameters
    W = tf.Variable(tf.zeros([number_of_pixels, 10]))
    # Initialize the structure used for the bias parameters
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(X, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean((-1.0 * tf.reduce_sum(y_ * tf.log(y),\
            reduction_indices=[1])))

    train = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    session = tf.Session()

    training_set_labels = dataset['training_labels']

    training_labels = np.zeros((len(training_set_labels), 10))

    for training_set_labels_index in range(len(training_set_labels)):
        one_hot = np.zeros((10), dtype=int)
        one_hot[int(training_set_labels[training_set_labels_index])] = 1
        training_labels[training_set_labels_index, :] = one_hot[:]

    session.run(init)

    print("training")
    for i in range(ITERATIONS):
        session.run(train, feed_dict={X : dataset['training_set'],\
                y_ : training_labels})

    cross_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

    test_set_labels = dataset['test_labels']

    test_labels = np.zeros((len(test_set_labels), 10))

    for test_set_labels_index in range(len(test_set_labels)):
        one_hot = np.zeros((10), dtype=int)
        one_hot[int(test_set_labels[test_set_labels_index])] = 1
        test_labels[test_set_labels_index, :] = one_hot[:]

    print(session.run(accuracy, feed_dict={X : dataset['test_set'],\
            y_ : test_labels}))
