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
from load_partial_dataset import loadImages, createBatchIndices

np.random.seed(1)

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

    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

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
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # Softmax activation is in cost function
    Z3 = tf.contrib.layers.fully_connected(P2, n_y, activation_fn=None)

    # new here
    #A3 = tf.nn.relu(Z3)
    #Z4 = tf.contrib.layers.fully_connected(A3, n_y, activation_fn=None)
    ### END CODE HERE ###

    return Z3


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    ### END CODE HERE ###

    return cost

def model(batches, dev_indices, Y_name, learning_rate = 0.01,
          num_epochs=5, print_cost=True):        #changed num Epochs to 2
    ops.reset_default_graph()
    tf.reset_default_graph()
    tf.set_random_seed(1)
    seed = 1

    n_H0, n_W0, n_C0 = (120, 160, 3)
    n_y = 8     # Change output size here
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters, n_y)
    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            seed = seed + 1
            train_accuracy = 0.0

            for i,batch in enumerate(batches):
                print("Batch:", i)
                X_batch, Y_batch = loadImages(batch, Y_name)
                X_batch = X_batch/255.
                Y_batch = np.asarray([list(y).index(1) for y in Y_batch])
                Y_batch = one_hot_matrix(Y_batch, C=n_y)
                _, temp_cost = sess.run([optimizer, cost],feed_dict={X: X_batch,Y: Y_batch})
                minibatch_cost += temp_cost / len(batches)
                #tf.Print(Z3, [Z3])
                predict_op = tf.argmax(Z3, 1)
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                train_accuracy += accuracy.eval({X: X_batch, Y: Y_batch}) / len(batches)
            print("Train Accuracy:", train_accuracy)

            # Print the cost every epoch
            if print_cost == True and epoch % 1 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

            # plot the cost
            #plt.plot(np.squeeze(costs))
            #plt.ylabel('cost')
            #plt.xlabel('iterations (per tens)')
            #plt.title("Learning rate =" + str(learning_rate))
            #plt.show()

        # Calculate the correct predictions
        print ("Calculating predictions now")
        predict_op = tf.argmax(Z3, 1)
        print("Equal calculation...")
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        print("Calculating train accuracy...")
        train_accuracy = 0
        for i, batch in enumerate(batches):
            print("Batch:", i)
            X_batch, Y_batch = loadImages(batch, Y_name)
            Y_batch = np.asarray([list(y).index(1) for y in Y_batch])
            Y_batch = one_hot_matrix(Y_batch, C=n_y)
            train_accuracy += accuracy.eval({X: X_batch, Y: Y_batch})/len(batches)
        print("Calculating dev accuracy...")
        X_dev, Y_dev = loadImages(dev_indices, Y_name)
        Y_dev = np.asarray([list(y).index(1) for y in Y_dev])
        Y_dev = one_hot_matrix(Y_dev, C=n_y)
        dev_accuracy = accuracy.eval({X: X_dev, Y: Y_dev})
        print("Train Accuracy:", train_accuracy)
        print("Dev Accuracy:", dev_accuracy)

        saver.save(sess, "./savedModels/cnn_test_model")

        return train_accuracy, dev_accuracy, parameters


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
    random.seed(1)

    NUM_IMAGES = 8400
    NUM_BATCHES = 10
    indices = np.random.permutation(np.arange(NUM_IMAGES))
    # splits into 80% train, 15% dev, and 5% test
    data_dividers = (0.8, 0.15, 0.05)
    train_limits = (0, int(data_dividers[0] * NUM_IMAGES))
    dev_limits = (int(data_dividers[0] * NUM_IMAGES),
                  int(data_dividers[0] * NUM_IMAGES + data_dividers[
                      1] * NUM_IMAGES))
    test_limits = \
        (int(data_dividers[0] * NUM_IMAGES + data_dividers[1] * NUM_IMAGES),
         NUM_IMAGES)

    train_inds = indices[train_limits[0]:train_limits[1]]
    dev_inds = indices[dev_limits[0]:dev_limits[1]]
    test_inds = indices[test_limits[0]:test_limits[1]]
    batches = createBatchIndices(train_inds, NUM_BATCHES)
    print(max(train_inds))
    print(max(dev_inds))
    print(max(test_inds))

    _, _, parameters = model(batches,dev_inds,'vectorY3.npy')
    W1 = parameters['W1']
    W2 = parameters['W2']

if __name__ == '__main__':
    main()