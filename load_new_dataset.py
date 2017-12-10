from PIL import Image, ImageFilter, ImageEnhance
import io
import os
import numpy as np
import glob
import cv2
import random
from create_images import loadImagesAndNames


'''
I already changed the size and mode of the images to the ones 
required in our models. So that function is in comments.
'''
def changeImageSize():

    realCards, names = loadImagesAndNames('/Users/matiascastillo/Desktop/CS229/Project/newImages/OtherImages/')

    print realCards
    newImages = []
    for i, image in enumerate(realCards):
        im = image.convert('RGB')
        ima = im.resize((640, 480), Image.ANTIALIAS)
        newImages.append(ima)

    for i, im in enumerate(newImages):
        directory = str('/Users/matiascastillo/Desktop/CS229/Project/newImages/resizedCards/'+str(names[i]))
        im.save(directory, 'JPEG')


def loadImagesX(indices, Y_vector):
    """
    Indices: indices of the training example to be used
    Y_vector: The entire Y vector of the training set
    """
    filenames = sorted(glob.glob('/Users/matiascastillo/Desktop/CS229/Project/newImages/resizedCards/*.JPG'))
    print(len(filenames))
    shape = (len(indices),) + np.asarray(cv2.imread(filenames[0])).shape
    X = np.zeros(shape)
    Y = np.zeros((len(indices),)+Y_vector[0].shape)
    for i,index in enumerate(indices):
        X[i] = cv2.imread(filenames[index])
        Y[i] = Y_vector[index]

    return X, Y

def main():
    arr = np.load('vectorY3X.npy')
    #changeImageSize()



if __name__ == '__main__':
    main()
