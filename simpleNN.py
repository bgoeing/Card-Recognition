import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import saver
from load_partial_dataset import loadImages, createBatchIndices

warnings.filterwarnings('error')
regularization = True

def display(x):
    matrix = np.reshape(x, (28, 28))
    plt.imshow(1 - matrix, cmap='Greys')
    plt.show()

def plot_line(trainArr, devArr, file_name, measure, epochs):
    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.set_xlabel('Epoch')
    sub.set_ylabel(measure)
    sub.plot(range(1, epochs + 1), trainArr)  # blue
    sub.plot(range(1, epochs + 1), devArr)  # orange
    plt.savefig(file_name)

def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def softmax(x):
    """
    Compute softmax function for input.
    Use tricks from previous assignment to avoid overflow
    """
    ### YOUR CODE HERE
    ex = np.exp(x - np.max(x))
    s = ex / np.sum(ex, axis=1, keepdims=True)
    ### END YOUR CODE
    return s

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    s = 1 / (1 + np.exp(-x))
    ### END YOUR CODE
    return s

def CELoss(y, label):
    return -math.log10(y[list(label).index(1)])

def batch_cost(y, labels):
    return sum(CELoss(y_i, label) for y_i, label in
               zip(y, labels)) / float(
        len(labels))

def initializeParams(features,outputs):
    num_hidden1 = 600
    num_hidden2 = 300
    params = {}
    params['W1'] = np.random.standard_normal(
        size=(features, num_hidden1))
    params['b1'] = np.zeros((1, num_hidden1))
    params['W2'] = np.random.standard_normal(
        size=(num_hidden1, num_hidden2))
    params['b2'] = np.zeros((1, num_hidden2))
    params['W3'] = np.random.standard_normal(
        size=(num_hidden2, outputs))
    params['b3'] = np.zeros((1, outputs))
    return params

def forward_prop(data, labels, params):
    """
    return hidden layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    W3 = params['W3']
    b3 = params['b3']

    ### YOUR CODE HERE
    Z1 = np.dot(data, W1) + b1
    a1 = sigmoid(Z1)
    Z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(Z2)
    Z3 = np.dot(a2, W3) + b3
    y = softmax(Z3)

    h1 = a1
    h2 = a2
    cost = batch_cost(y, labels)
    ### END YOUR CODE
    return h1, h2, y, cost

def backward_prop(data, labels, params):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    W3 = params['W3']
    b3 = params['b3']

    ### YOUR CODE HERE
    gradW1 = np.zeros(W1.shape)
    gradW2 = np.zeros(W2.shape)
    gradW3 = np.zeros(W3.shape)
    gradb1 = np.zeros(b1.shape)
    gradb2 = np.zeros(b2.shape)
    gradb3 = np.zeros(b3.shape)

    h1, h2, y, cost = forward_prop(data, labels, params)

    lambda_ = 0.0002
    global regularization
    if regularization:
        regW3 = 2 * lambda_ * W3
        regW2 = 2 * lambda_ * W2
        regW1 = 2 * lambda_ * W1
    else:
        regW3 = 0
        regW2 = 0
        regW1 = 0

    delta3 = (y - labels) / len(data)
    gradb3 = np.sum(delta3, axis=0)
    gradW3 = np.dot(np.transpose(h2), delta3) + regW3
    delta2 = np.dot(delta3, np.transpose(W3)) * h2 * (1 - h2)
    gradb2 = np.sum(delta2, axis=0)
    gradW2 = np.dot(np.transpose(h1), delta2) + regW2
    delta1 = np.dot(delta2, np.transpose(W2)) * h1 * (1 - h1)
    gradb1 = np.sum(delta1, axis=0)
    gradW1 = np.dot(np.transpose(data), delta1) + regW1
    ### END YOUR CODE

    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['W3'] = gradW3
    grad['b1'] = gradb1
    grad['b2'] = gradb2
    grad['b3'] = gradb3

    return grad

def nn_train(batches,dev_indices,Y_name, learning_rate=0.1,num_epochs = 10,print_cost=True, start_params = None):
    n_H0, n_W0, n_C0 = (120, 160, 3)
    # n_H0, n_W0, n_C0 = (480, 640, 3)
    n_y = 52
    features = n_H0*n_W0*n_C0
    outputs = n_y

    ### YOUR CODE HERE
    if start_params == None:
        params = initializeParams(features,outputs)
    else:
        params = start_params

    trainLoss = [0 for e in range(num_epochs)]
    devLoss = [0 for e in range(num_epochs)]
    trainAcc = [0 for e in range(num_epochs)]
    devAcc = [0 for e in range(num_epochs)]

    for e in range(num_epochs):
        learning_rate = learning_rate * (10 * num_epochs - e) / (
        10 * num_epochs)
        print('learning rate: {0}'.format(learning_rate))
        for b, batch in enumerate(batches):
            print('epoch: {0}, batch: {1}'.format(e + 1,
                                                  b + 1))

            X_batch, Y_batch = loadImages(batch, Y_name)
            X_batch = X_batch / 255
            X_batch = np.reshape(X_batch,(len(X_batch),n_H0*n_W0* n_C0))
            Y_batch = np.asarray([list(y).index(1) for y in Y_batch])
            Y_batch = one_hot_labels(Y_batch, C=n_y)

            grad = backward_prop(X_batch, Y_batch,
                                 params)
            params['W1'] -= learning_rate * grad['W1']
            params['W2'] -= learning_rate * grad['W2']
            params['W3'] -= learning_rate * grad['W3']
            params['b1'] -= learning_rate * grad['b1']
            params['b2'] -= learning_rate * grad['b2']
            params['b3'] -= learning_rate * grad['b3']

        print("Calculating loss...")
        for b, batch in enumerate(batches):
            X_batch, Y_batch = loadImages(batch, Y_name)
            X_batch = np.reshape(X_batch,
                                 (len(X_batch), n_H0 * n_W0 * n_C0))
            X_batch = X_batch/255
            Y_batch = np.asarray([list(y).index(1) for y in Y_batch])
            Y_batch = one_hot_labels(Y_batch, C=n_y)

            trainLoss[e] += forward_prop(X_batch, Y_batch, params)[3]/len(batches)
            trainAcc[e] += nn_test(X_batch, Y_batch, params)/len(batches)
        X_dev, Y_dev = loadImages(dev_indices, Y_name)
        X_dev = np.reshape(X_dev,
                             (len(X_dev), n_H0 * n_W0 * n_C0))
        X_dev = X_dev / 255
        Y_dev = np.asarray([list(y).index(1) for y in Y_dev])
        Y_dev = one_hot_labels(Y_dev, C=n_y)
        devLoss[e] = forward_prop(X_dev, Y_dev, params)[3]
        devAcc[e] = nn_test(X_dev, Y_dev, params)

        del X_dev
        del Y_dev
        if print_cost:
            print("Train cost after epoch %i: %f" % (e+1, trainLoss[e]))
            print("Train accuracy after epoch %i: %f" % (e+1, trainAcc[e]))
        #print("Dev cost after epoch %i: %f" % (e, devLoss[e]))
    plot_line(trainLoss, devLoss,'loss.png','Loss',epochs=num_epochs)
    plot_line(trainAcc, devAcc,'accuracy.png','Accuracy',epochs=num_epochs)
    ### END YOUR CODE

    return params

def nn_test(data, labels, params):
    h1,h2, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    truthVector = np.argmax(output, axis=1) == np.argmax(
        labels, axis=1)
    accuracy = truthVector.sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels,C):
    one_hot_labels = np.zeros((labels.size, C))
    one_hot_labels[
        np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels

def main():
    np.random.seed(100)
    """
    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std
    """
    NUM_IMAGES = 54600
    NUM_BATCHES = 30
    NUM_EPOCHS = 30
    indices = np.random.permutation(np.arange(NUM_IMAGES))
    # splits into 90% train, 5% dev, and 5% test
    data_dividers = (0.9, 0.05, 0.05)
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

    print('{0} examples'.format(len(train_inds)))
    print('{0} batches'.format(len(batches)))

    params = nn_train(batches,dev_inds,'vectorY4.npy',num_epochs=NUM_EPOCHS)
    saver.saveParameters(params,'test.txt')

    readyForTesting = False
    if readyForTesting:
        print('Testing...')
        X_test, Y_test = loadImages(test_inds, 'vectorY1.npy')
        n_H0 , n_W0 , n_C0 = X_test[0].shape
        X_test = X_test/255
        X_test = np.reshape(X_test,
                           (len(X_test), n_H0 * n_W0 * n_C0))
        Y_test = np.asarray([list(y).index(1) for y in Y_test])
        Y_test = one_hot_labels(Y_test, C=52)
        accuracy = nn_test(X_test, Y_test, params)
        print('Test accuracy: %f' % accuracy)

if __name__ == '__main__':
    main()
