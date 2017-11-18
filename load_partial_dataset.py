import numpy as np
import random
import glob
import cv2

randoms = [1630, 777, 3000, 5247, 2519, 1991, 7254, 3333, 6436, 3368, 6672, 737, 479, 701, 3598, 4467, 933, 976, 4393, 7076, 6581, 2716, 972, 5613, 5774, 7164, 715, 845, 4090, 524, 6807, 2863, 1846, 6102, 2644, 2046, 7981, 5927, 1445, 4724, 951, 497, 8399, 8024, 1871, 6287, 6027, 7820, 6868, 2779, 2472, 7997, 6326, 2678, 3075, 2567, 3513, 15, 6333, 1337, 7687, 2327, 8329, 4957, 2776, 6687, 6921, 3300, 6659, 6047, 2583, 1838, 4640, 2777, 1661, 8116, 710, 5454, 3322, 7162, 2515, 2669, 7721, 7609, 6175, 6602, 3860, 2668, 2218, 5193, 6106, 6558, 1837, 1722, 4277, 3381, 3559, 6225, 1825, 5664]

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
    arr = np.load('vectorY1.npy')
    print(loadImages(random,arr))


if __name__ == '__main__':
    main()