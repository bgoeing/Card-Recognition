import numpy as np
import random
import glob
import cv2
import timeit

def loadImages(indices, Y_name):
    """
    Indices: indices of the training example to be used
    Y_vector: The entire Y vector of the training set
    """
    Y_vector = np.load(Y_name)
    filenames = sorted(glob.glob('created_images/*.JPG'))
    shape = (len(indices),) + np.asarray(cv2.imread(filenames[0])).shape
    X = np.zeros(shape)
    Y = np.zeros((len(indices),) + Y_vector[0].shape)
    for i,index in enumerate(indices):
        X[i] = cv2.imread(filenames[index])
        Y[i] = Y_vector[index]
    return X,Y


def loadImage(name, Y_name):
    """
    Indices: indices of the training example to be used
    Y_vector: The entire Y vector of the training set
    """
    Y_vector = np.load(Y_name)
    filenames = sorted(glob.glob('generatedCards/*.JPG'))
    shape = (1,) + np.asarray(cv2.imread(filenames[0])).shape
    X = np.zeros(shape)
    Y = np.zeros((1,) + Y_vector[0].shape)
    X[0] = cv2.imread('generatedCards/'+name)
    Y[0] = Y_vector[filenames.index('generatedCards/'+name)]
    return X,Y

def loadNewImage():
    """
    Indices: indices of the training example to be used
    Y_vector: The entire Y vector of the training set
    """
    img = cv2.imread('img.jpg')
    shape = (1,) + np.asarray(img).shape
    X = np.zeros(shape)
    Y = np.zeros((1,52))
    X[0] = img
    y = [0 for i in range(52)]
    y[19] = 1
    Y[0] = np.array(y)
    return X,Y

def loadOtherImages():
    Y_vector = np.load('newVectorY52.npy')
    filenamesJPG = glob.glob('generatedNewCards/*.jpeg')
    filenamesPNG = glob.glob('generatedNewCards/*.png')
    filenames = sorted(filenamesJPG + filenamesPNG)
    shape = (len(filenames),) + np.asarray(cv2.imread(filenames[0])).shape
    X = np.zeros(shape)
    Y = np.zeros((len(filenames),) + Y_vector[0].shape)
    for i,name in enumerate(filenames):
        X[i] = cv2.imread(name)
        Y[i] = Y_vector[i]
    return X,Y

def createBatchIndices(indices,num_batches):
    """
    Divides indices into n equally large lists
    Assumes that indices are already randomized
    Last batch may be at most num_batches smaller than the other batches
    """
    m = len(indices)
    batches = []
    batch_size = int(np.ceil(m/num_batches))
    for i in range(num_batches):
        batches.append(indices[i*batch_size:(i+1)*batch_size])
    return batches

def main():
    NUM_IMAGES = 8400
    NUM_BATCHES = 10
    indices = np.random.permutation(np.arange(NUM_IMAGES))
    data_dividers = (0.8,0.15,0.05)
    train_limits = (0,int(data_dividers[0]*NUM_IMAGES))
    dev_limits = (int(data_dividers[0]*NUM_IMAGES),
        int(data_dividers[0] * NUM_IMAGES + data_dividers[1]*NUM_IMAGES))
    test_limits = \
        (int(data_dividers[0] * NUM_IMAGES + data_dividers[1]*NUM_IMAGES),
         NUM_IMAGES)

    train_inds = indices[train_limits[0]:train_limits[1]]
    dev_inds = indices[dev_limits[0]:dev_limits[1]]
    test_inds = indices[test_limits[0]:test_limits[1]]
    batches = createBatchIndices(train_inds, NUM_BATCHES)

    # Test: loading all batches, one after another
    for i in range(len(batches)):
        print('Batch {0}'.format(i+1))
        start = timeit.default_timer()
        X,Y = loadImages(batches[i],'vectorY1.npy')
        end = timeit.default_timer()
        print('Time taken: {0}'.format(end - start))
    print('finished')

if __name__ == '__main__':
    main()