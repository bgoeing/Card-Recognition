import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
import cv2
import glob
from numpy.random import multinomial
import random
from load_partial_dataset import loadImages, createBatchIndices, \
    loadImage, loadOtherImages

import os, utils

#np.random.seed(1)

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    ### START CODE HERE ### (≈2 lines)
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))
    ### END CODE HERE ###

    return X, Y

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [4, 4, 3, 8],
                         initializer=tf.contrib.layers.xavier_initializer(
                             seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16],
                         initializer=tf.contrib.layers.xavier_initializer(
                             seed=0))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters

def forward_propagation(X, parameters, n_y):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1],
                        padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    tf.add_to_collection('conv1_output', A1)
    tf.add_to_collection('conv2_output', A2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1],
                        padding='SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # Softmax activation is in cost function
    Z3 = tf.contrib.layers.fully_connected(P2, n_y, activation_fn=None)

    # new here
    # A3 = tf.nn.relu(Z3)
    # Z4 = tf.contrib.layers.fully_connected(A3, n_y, activation_fn=None)
    ### END CODE HERE ###

    return Z3

PLOT_DIR = './'

def plot_conv_output(conv_img, name):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    Source : https://github.com/grishasergei/conviz/blob/master/conviz.py
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_output')
    plot_dir = os.path.join(plot_dir, name)

    print(conv_img.shape)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :,  l]
        # put it on the grid
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')

def getPredictions(y):
    output = []
    for i in y:
        name = ''
        suit = i//13
        if suit == 0:
            name += 'clubs '
        elif suit == 1:
            name += 'diamonds '
        elif suit == 2:
            name += 'hearts '
        elif suit == 3:
            name += 'spades '
        rank = (i%13)+1
        name += str(rank)
        output.append(name)
    return output

def model(batches, dev_indices, test_indices, Y_name, learning_rate=0.01,
          num_epochs=5, print_cost=True):  # changed num Epochs to 2
    ops.reset_default_graph()
    tf.reset_default_graph()
    tf.set_random_seed(1)

    n_H0, n_W0, n_C0 = (160, 120, 3)
    n_y = 52  # Change output size here

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters, n_y)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, './newSavedModels/cnn_test_model')
        print("Model restored from file: %s" % './newSavedModels/cnn_test_model')


        # Calculate the correct predictions
        print("Calculating predictions now")
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracies
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        X_dev, Y_dev = loadImages(dev_indices, Y_name)
        dev_accuracy = accuracy.eval({X: X_dev, Y: Y_dev})

        print("Dev Accuracy: {0}".format(dev_accuracy))

        card_name = 'hearts-6-3-zoom0.2.JPG'
        X_card, Y_card = loadImage(card_name,Y_name)
        card_accuracy = accuracy.eval({X: X_card, Y: Y_card})

        print("Card ({0}) Accuracy: {1}".format(card_name,card_accuracy))

        X_other, Y_other = loadOtherImages()
        other_accuracy = accuracy.eval({X: X_other, Y: Y_other})

        print("Other Accuracy: {0}".format(other_accuracy))

        y = predict_op.eval(feed_dict={X: X_other})
        print(y)
        names = getPredictions(y)
        filenamesJPG = glob.glob('generatedNewCards/*.jpeg')
        filenamesPNG = glob.glob('generatedNewCards/*.png')
        filenames = sorted(filenamesJPG + filenamesPNG)
        for i,name in enumerate(names):
            print('{0} == {1}'.format(filenames[i],name))
        return

        conv1_out = sess.run([tf.get_collection('conv1_output')],
                            feed_dict={X: X_card})
        for i, c in enumerate(conv1_out[0]):
            plot_conv_output(c, 'conv1')
        conv2_out = sess.run([tf.get_collection('conv2_output')],
                             feed_dict={X: X_card})
        for i, c in enumerate(conv2_out[0]):
            plot_conv_output(c, 'conv2')
        print('done')

        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()
        return parameters


def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """

    ### START CODE HERE ###

    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name="C")
    one_hot_matrix = tf.one_hot(labels, C, 1)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    ### END CODE HERE ###

    return one_hot


def main():
    #random.seed(1)

    NUM_IMAGES = 54600
    NUM_BATCHES = 1
    indices = np.random.permutation(np.arange(NUM_IMAGES))
    # splits into 90% train, 5% dev, and 5% test
    data_dividers = (0.05, 0.05, 0.9)
    train_limits = (0, int(data_dividers[0] * NUM_IMAGES))
    dev_limits = (int(data_dividers[0] * NUM_IMAGES),
                  int(data_dividers[0] * NUM_IMAGES + data_dividers[
                      1] * NUM_IMAGES))
    test_limits = \
        (int(data_dividers[0] * NUM_IMAGES + data_dividers[1] * NUM_IMAGES),
         NUM_IMAGES)

    train_inds = indices[train_limits[0]:train_limits[1]]
    train_inds = train_inds[:500]
    dev_inds = indices[dev_limits[0]:dev_limits[1]][:300]
    test_inds = indices[test_limits[0]:test_limits[1]]
    batches = createBatchIndices(train_inds, NUM_BATCHES)

    model(batches, dev_inds, test_inds, 'vectorY52.npy')


if __name__ == '__main__':
    main()