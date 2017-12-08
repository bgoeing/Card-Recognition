import numpy as np
import random
import glob
import cv2
import timeit

def loadImages(indices):
    """
    Indices: indices of the training example to be used
    Y_vector: The entire Y vector of the training set
    """
    print('loading Y vectors')
    Y1_vector = np.load('vectorY1.npy')
    Y2_vector = np.load('vectorY2.npy')
    print('getting names')
    filenames = sorted(glob.glob('generatedCards/*.JPG'))
    shape = (len(indices),) + np.asarray(cv2.imread(filenames[0])).shape
    X = np.zeros(shape)
    Y1 = np.zeros((len(indices),) + Y1_vector[0].shape)
    Y2 = np.zeros((len(indices),) + Y2_vector[0].shape)
    print('reading images')
    for i,index in enumerate(indices):
        X[i] = cv2.imread(filenames[index])
        Y1[i] = Y1_vector[index]
        Y2[i] = Y2_vector[index]
    print('done')
    return X,Y1,Y2

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
        X,Y1,Y2 = loadImages(batches[i])
        end = timeit.default_timer()
        print('Time taken: {0}'.format(end - start))
    print('finished')

if __name__ == '__main__':
    main()