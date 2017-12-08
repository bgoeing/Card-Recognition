import numpy as np
import random
import glob
import cv2


def loadImages(indices, Y_vector):
    """
    Indices: indices of the training example to be used
    Y_vector: The entire Y vector of the training set
    """
    filenames = sorted(glob.glob('generatedCards/*.JPG'))
    shape = (len(indices),) + np.asarray(cv2.imread(filenames[0])).shape
    X = np.zeros(shape)
    Y = np.zeros((len(indices),)+Y_vector[0].shape)
    for i,index in enumerate(indices):
        X[i] = cv2.imread(filenames[index])
        Y[i] = Y_vector[index]

    return X,Y

def main():
    arr = np.load('vectorY3.npy')
    print(loadImages(random,arr))


if __name__ == '__main__':
    main()
