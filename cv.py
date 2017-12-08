import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
# from cnn_utils import *
import cv2
import glob
from numpy.random import multinomial
import random
from load_partial_dataset import loadImages


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

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 2 lines of code)
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


def forward_propagation(X, parameters):
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
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(P2, 4, activation_fn=None)
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


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=15, minibatch_size=64, print_cost=True):        #changed num Epochs to 2
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3

    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1

            """
            minibatches = random_mini_batches(X_train, Y_train, 4, minibatch_size, seed)
            
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches
            """

            _, temp_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            minibatch_cost += temp_cost

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
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        saver.save(sess, "./savedModels/cnn_test_model")

        return train_accuracy, test_accuracy, parameters


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

    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, C, 1)

    # Create the session (approx. 1 line)
    sess = tf.Session()

    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)
    #one_hot = one_hot.T

    # Close the session (approx. 1 line). See method 1 above.
    sess.close()

    ### END CODE HERE ###

    return one_hot

def main():
    random.seed(1)
    random_train = [random.randint(0, 8400) for i in range(350)]
    random_dev = [random.randint(0, 8400) for i in range(100)]

    train_full = [random.randint(0, 8400) for i in range(5000)]

    arr = np.load('vectorY1.npy')

    X_train, Y = loadImages(random_train, arr)
    X_test, Y_ = loadImages(random_dev, arr)

    Y_values = [list(y).index(1) for y in Y]
    Y_train = np.asarray(Y_values)
    #Y_train = Y_values.reshape(Y_values.shape[0], -1)

    Y_testvalues = [list(y).index(1) for y in Y_]
    Y_test = np.asarray(Y_testvalues)
    #Y_test = Y_testvalues.reshape(Y_testvalues.shape[0], -1)

    del Y, Y_values
    del Y_, Y_testvalues

    Y_train = one_hot_matrix(Y_train, C=4)
    Y_test = one_hot_matrix(Y_test, C=4)

    print(Y_train.shape)
    print(Y_test.shape)

    _, _, parameters = model(X_train, Y_train, X_test, Y_test)
    W1 = parameters['W1']
    W2 = parameters['W2']

    """

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    v = sess.run(W1)
    k = sess.run(W2)

    #print(type(tf.Session().run(tf.constant([1, 2, 3]))))
    """


    """

    a = []
    im1 = cv2.imread("pic1.jpeg")
    im2 = cv2.imread("pic2.jpeg")
    # arr = np.append(im1, im2)
    a.append(im1)
    a.append(im2)
    a = np.asarray(a)
    """

    """
    a = []
    im1 = cv2.imread("pic1.jpeg")
    im2 = cv2.imread("pic2.jpeg")
    # arr = np.append(im1, im2)
    a.append(im1)
    a.append(im2)
    a = np.asarray(a)
    Y_all_orig = np.load("numpyVectorY1.npy")
    print(Y_all_orig[0:2])
    K = Y_all_orig[0:2] / 255.


    print (K.shape)
    print(a.shape)


    k = np.load("numpyVectorY1.npy")
    print(k.shape)

    print(type(im1))
    print(im1.shape)
    print(a.shape)
    print(k[1])
    """
    """
    #filenames = glob.glob('testff/*.JPG')
    filenames = glob.glob('generatedCards/*.JPG')
    i = 0
    X_all_orig = []
    for filename in filenames:
        X_all_orig.append(cv2.imread(filename))
        print (i)
        i+=1
    X_all_orig = np.asarray(X_all_orig)
    print("here0")
    np.save("X_all_orig", X_all_orig)
    print("here1")
    #X_all_orig = np.load("X_all_orig.npy")
    Y_all_orig = np.load("numpyVectorY1.npy")

    print ("here2")
    X_all =  X_all_orig / 255.
    Y_all = Y_all_orig / 255.

    #idx = np.random.choice(np.arange(len(8400)), 800, replace=False)
    #x_sample = X_all[idx]
    #y_sample = Y_all[idx]

    print (X_all.shape)
    print (Y_all.shape)
    #print (x_sample.shape)
    #print (y_sample.shape)
    """

    """

    print (X_all.shape)
    print (Y_all.shape)



    # X = np.genfromtxt('logistic_x.txt')

    # Create dummy placeholders
    X, Y = create_placeholders(64, 64, 3, 6)
    print("X = " + str(X))
    print("Y = " + str(Y))

    # Initialize paramters
    tf.reset_default_graph()
    with tf.Session() as sess_test:
        parameters = initialize_parameters()
        init = tf.global_variables_initializer()
        sess_test.run(init)
        print("W1 = " + str(parameters["W1"].eval()[1, 1, 1]))
        print("W2 = " + str(parameters["W2"].eval()[1, 1, 1]))


    #Forward prop

    tf.reset_default_graph()

    with tf.Session() as sess:
        np.random.seed(1)
        X, Y = create_placeholders(64, 64, 3, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        init = tf.global_variables_initializer()
        sess.run(init)
        a = sess.run(Z3, {X: np.random.randn(2, 64, 64, 3), Y: np.random.randn(2, 6)})
        print("Z3 = " + str(a))


    # Compute Cost
    tf.reset_default_graph()

    with tf.Session() as sess:
        np.random.seed(1)
        X, Y = create_placeholders(64, 64, 3, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        cost = compute_cost(Z3, Y)
        init = tf.global_variables_initializer()
        sess.run(init)
        a = sess.run(cost, {X: np.random.randn(4, 64, 64, 3), Y: np.random.randn(4, 6)})
        print("cost = " + str(a))

    # testfile = np.loadtxt("vectorY1.txt")
    testfile = np.loadtxt("mess.txt", delimiter=',', dtype='int')
    print(testfile)
    # print (testfile)
    print(type(testfile))
    # Y_train = convert_to_one_hot(testfile, 4).T
    Y_train = tf.one_hot(testfile, 4)
    print(Y_train[2])

    # b = np.zeros((4, 4))
    # b[np.arange(4), testfile] = 1
    # print(b)

    # testfile = np.loadtxt("vectorY1.txt", delimiter=',')
    # testfile = open("vectorY1.txt")
    print(testfile)
"""


if __name__ == '__main__':
    main()