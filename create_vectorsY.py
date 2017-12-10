from PIL import Image, ImageFilter, ImageEnhance
import io
import os
import numpy as np
import operator

'''
This file loads images from a given folder, creates a new set of augmented images,
and saves them in another folder.
'''

def loadImagesAndNames(directory):
    realImages = []
    names = [name for name in os.listdir(directory) if name != ".DS_Store"]
    sortedNames = orderFileNames(names)
    return realImages, sortedNames

def orderFileNames(names):
    NameList = []
    for name in names:
        #if name[0] == 'c':
        newName = name.split('-')
        newName[1] = int(newName[1])
        newName[2] = int(newName[2])
        NameList.append(newName)
    NameList.sort(key = lambda row: (row[0],row[1], row[2]))

    sortedNames = []
    for name in NameList:
        name[1] = str(name[1])
        name[2] = str(name[2])
        newName = '-'.join(name)
        sortedNames.append(newName)
    return sortedNames

def getVectorY4(cardName):
    if cardName[0] == 'c':
        return [1, 0, 0, 0]
    if cardName[0] == 'd':
        return [0, 1, 0, 0]
    if cardName[0] == 'h':
        return [0, 0, 1, 0]
    if cardName[0] == 's':
        return [0, 0, 0, 1]

def getVectorY52(cardName):
    number = cardName.split('-')
    vector = [0]*52
    position = int(number[1])
    if cardName[0] == 'c':
        vector[position-1] = 1
    if cardName[0] == 'd':
        vector[position-1+13] = 1
    if cardName[0] == 'h':
        vector[position-1+26] = 1
    if cardName[0] == 's':
        vector[position-1+39] = 1
    return vector

def getVectorY13(cardName):
    number = cardName.split('-')
    vector = [0]*13
    position = int(number[1])
    vector[position-1] = 1
    return vector

def getVectorY8(cardName):
    number = cardName.split('-')
    vector = [0]*8
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


def saveNumpyVectors(vectorY52, vectorY13, vectorY4):

    VectorY52 = np.array([np.array(x) for x in vectorY52])
    VectorY13 = np.array([np.array(x) for x in vectorY13])
    VectorY4 = np.array([np.array(x) for x in vectorY4])

    np.save('vectorY52.npy', VectorY52, fix_imports = True)
    np.save('vectorY13.npy', VectorY13, fix_imports = True)
    np.save('vectorY4.npy', VectorY4, fix_imports = True)


def saveTextVectors(vectorY52, vectorY13, vectorY4):
    with io.FileIO("vectorY52.txt", "w") as file1:
        for y in vectorY52:
            file1.writelines(str(y))
        file1.close()
    with io.FileIO("vectorY13.txt", "w") as file2:
        for y in vectorY13:
            file2.writelines(str(y))
        file2.close()
    with io.FileIO("vectorY4.txt", "w") as file3:
        for y in vectorY4:
            file3.writelines(str(y))
        file3.close()


'''generate vectorY1 mx4 for (diamonds, hearts, spades, clubs)
   generate vectorY2 mx13 for (As, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K)'''
def generateVectorsY(images, names):
    vectorY52 = []
    vectorY13 = []
    vectorY4 = []
    for name in names:
        vectorY52.append(getVectorY52(name))
        vectorY13.append(getVectorY13(name))
        vectorY4.append(getVectorY4(name))
    return vectorY52, vectorY13, vectorY4

'''
The vectors Y1 and Y2 are saved in the same folder as this python file.
The vectors are saved in both .txt and .npy formats.
You must specify here the directory of the folder containing the images
from which the vectorsY will be generated.
'''
def main():

    CardsDirectory = '/Users/matiascastillo/Desktop/CS229/Project/Card-Recognition/created_images/'
    cards, names = loadImagesAndNames(CardsDirectory)
    vectorY52, vectorY13, vectorY4 = generateVectorsY(cards, names)

    saveNumpyVectors(vectorY52, vectorY13, vectorY4)
    saveTextVectors(vectorY52, vectorY13, vectorY4)

    return

if __name__ == '__main__':
    main()
