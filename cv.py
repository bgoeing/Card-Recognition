import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
#from cnn_utils import *
import cv2
import glob
from numpy.random import multinomial



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
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
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
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)
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

def loadImages():
    filenames = glob.glob('generatedCards/*.JPG')
    i = 0
    X_all_orig = []
    for filename in filenames:
        X_all_orig.append(cv2.imread(filename))
        print (i)
        i+=1
    X_all_orig = np.asarray(X_all_orig)
    return X_all_orig

def createTTDSets():

    X_all_orig = np.load("X_all.npy")
    Y_all = np.load("numpyVectorY1.npy")
    norm = np.float16(255.)
    X_all = X_all_orig / norm

    print("success so far")

    #a1, a2, a3 = np.vsplit(Y_all[np.random.permutation(Y_all.shape[0])], (7400, 8000))

    n_total_samples = 8400  # make dynamic

    indices = np.arange(n_total_samples)
    inds_split = multinomial(n=1,
                             pvals=[0.8, 0.15, 0.05],
                             size=n_total_samples).argmax(axis=1)

    train_inds = indices[inds_split == 0]
    test_inds = indices[inds_split == 1]
    dev_inds = indices[inds_split == 2]

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_dev = []
    Y_dev = []

    for ind in dev_inds:
        X_dev.append(X_all[ind])
        Y_dev.append(Y_all[ind])

    print("Stop1")

    for ind in test_inds:
        X_test.append(X_all[ind])
        Y_test.append(Y_all[ind])

    print("Stop2")

    for ind in train_inds:
        X_train.append(X_all[ind])
        Y_train.append(Y_all[ind])

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)
    X_dev = np.asarray(X_dev)
    Y_dev = np.asarray(Y_dev)

    print("Stop3")


    print("dev done so far")
    print(X_dev.shape)
    print(Y_dev.shape)

    np.save("X_train", X_train)
    np.save("Y_train", Y_train)

    np.save("X_test", X_test)
    np.save("Y_test", Y_test)

    np.save("X_dev", X_dev)
    np.save("Y_dev", Y_dev)


    print ("lololol")
    return 5

def main():
    """

    a = []
    im1 = cv2.imread("pic1.jpeg")
    im2 = cv2.imread("pic2.jpeg")
    # arr = np.append(im1, im2)
    a.append(im1)
    a.append(im2)
    a = np.asarray(a)
    """

    print("hi")
    n = createTTDSets()
    print (n)


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