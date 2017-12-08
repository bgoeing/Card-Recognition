from PIL import Image, ImageFilter, ImageEnhance
import io
import os
import numpy as np

'''
This file loads images from a given folder, creates a new set of augmented images,
and saves them in another folder.
'''

def loadImagesAndNames(directory):
    realImages = []
    names = [name for name in os.listdir(directory) if name != ".DS_Store"]
    for item in names:
        if os.path.isfile(directory+item):
            im=Image.open(directory+item)
            realImages.append(im)
            im.close()
    return realImages, names

def getVectorY1(cardName):
    if cardName[0] == 'd':
        return [1, 0, 0, 0]
    if cardName[0] == 'h':
        return [0, 1, 0, 0]
    if cardName[0] == 's':
        return [0, 0, 1, 0]
    if cardName[0] == 'c':
        return [0, 0, 0, 1]

def getVectorY2(cardName):
    number = cardName.split('-')
    vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    position = int(number[1])
    vector[position-1] = 1
    return vector

def getVectorY3(cardName):
    number = cardName.split('-')
    vector = [0, 0, 0, 0, 0, 0, 0, 0]
    position = int(number[1])
    if position == 1:
        if cardName[0] == 'd':
            return [1, 0, 0, 0, 0, 0, 0, 0]
        if cardName[0] == 'h':
            return [0, 1, 0, 0, 0, 0, 0, 0]
        if cardName[0] == 's':
            return [0, 0, 1, 0, 0, 0, 0, 0]
        if cardName[0] == 'c':
            return [0, 0, 0, 1, 0, 0, 0, 0]
    if position == 6:
        if cardName[0] == 'd':
            return [0, 0, 0, 0, 1, 0, 0, 0]
        if cardName[0] == 'h':
            return [0, 0, 0, 0, 0, 1, 0, 0]
        if cardName[0] == 's':
            return [0, 0, 0, 0, 0, 0, 1, 0]
        if cardName[0] == 'c':
            return [0, 0, 0, 0, 0, 0, 0, 1]

        return vector


def saveNumpyVectors(vectorY1, vectorY2, vectorY3):

    VectorY1 = np.array([np.array(x) for x in vectorY1])
    VectorY2 = np.array([np.array(x) for x in vectorY2])
    VectorY3 = np.array([np.array(x) for x in vectorY3])

    np.save('vectorY1X.npy', VectorY1, fix_imports = True)
    np.save('vectorY2X.npy', VectorY2, fix_imports = True)
    np.save('vectorY3X.npy', VectorY2, fix_imports = True)


def saveTextVectors(vectorY1, vectorY2, vectorY3):
    with io.FileIO("vectorY1X.txt", "w") as file1:
        for y in vectorY1:
            file1.writelines(str(y))
        file1.close()
    with io.FileIO("vectorY2X.txt", "w") as file2:
        for y in vectorY2:
            file2.writelines(str(y))
        file2.close()
    with io.FileIO("vectorY3X.txt", "w") as file3:
        for y in vectorY3:
            file3.writelines(str(y))
        file3.close()


'''generate vectorY1 mx4 for (diamonds, hearts, spades, clubs)
   generate vectorY2 mx13 for (As, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K)'''
def generateVectorsY(images, names):
    vectorY1 = []
    vectorY2 = []
    for name in names:
        vectorY1.append(getVectorY1(name))
        vectorY2.append(getVectorY2(name))
    return vectorY1, vectorY2

def generateVectorY3(images, names):
    vectorY3 = []
    for name in names:
        vectorY3.append(getVectorY3(name))
    return vectorY3

'''
The vectors Y1 and Y2 are saved in the same folder as this python file.
The vectors are saved in both .txt and .npy formats.
You must specify here the directory of the folder containing the images
from which the vectorsY will be generated.
'''
def main():

    CardsDirectory = '/Users/matiascastillo/Desktop/CS229/Project/newImages/OtherImages/'
    cards, names = loadImagesAndNames(CardsDirectory)
    vectorY1, vectorY2 = generateVectorsY(cards, names)
    vectorY3 = generateVectorY3(cards, names)
    saveNumpyVectors(vectorY1, vectorY2, vectorY3)
    saveTextVectors(vectorY1, vectorY2, vectorY3)

    return

if __name__ == '__main__':
    main()
