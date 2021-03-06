from PIL import Image, ImageFilter, ImageEnhance
import sys,os

'''
This file loads images from a given folder, creates a new set of augmented images,
and saves them in another folder.
'''

def loadImagesAndNames(directory):
    realImages = []
    names = [name for name in os.listdir(directory) if name != ".DS_Store"]
    for item in names:
        if os.path.isfile(directory+item):
            image = Image.open(directory+item)
            if hasattr(image, '_getexif'):
                orientation = 0x0112
                exif = image._getexif()
                if exif is not None:
                    orientation = exif[orientation]
                    rotations = {
                        3: Image.ROTATE_180,
                        6: Image.ROTATE_270,
                        8: Image.ROTATE_90
                    }
                    if orientation in rotations:
                        image = image.transpose(rotations[orientation])
            realImages.append(image)
    return realImages, names

def rotateImages(images):
    # rotate them 90 degrees to make them vertical.
    rotatedImages = []
    for image in images:
        rotatedImage = image.rotate(90)
        rotatedImages.append(rotatedImage)
    return rotatedImages

def scaleImages(images):
    # scale the images: 480x640 -> 120x160
    size = [120,160]
    scaledImages = []
    for image in images:
        scaledImage = image.resize(size, Image.ANTIALIAS)
        scaledImages.append(scaledImage)
    return scaledImages

#for a image, return contrasted images.
def contrast(image, name, contrastRate, numberImages):
    contrastedImages = []
    contraster = ImageEnhance.Contrast(image)
    i = 1.0 - contrastRate
    while len(contrastedImages) < numberImages:
        contrastedImages.append((contraster.enhance(i), createNewName(name, i, 'contrast')))
        i = i - contrastRate
    return contrastedImages

def color(image, name, colorRate, numberImages):
    colorImages = []
    colorer = ImageEnhance.Color(image)
    i = 1.0 - colorRate
    while len(colorImages) < numberImages:
        colorImages.append((colorer.enhance(i), createNewName(name, i, 'color')))
        i = i - colorRate
    return colorImages

def brightness(image, name, brightRate, numberImages):
    brightImages = []
    brighter = ImageEnhance.Brightness(image)
    i = 1.0 - brightRate
    while len(brightImages) < numberImages:
        brightImages.append((brighter.enhance(i), createNewName(name, i, 'bright')))
        i = i - brightRate
    return brightImages

# from 1 to 2 in more sharp, from 1 to 0 is more blurry.
def sharpness(image, name, sharpRate, numberImages):
    sharpImages = []
    sharpness = ImageEnhance.Sharpness(image)
    i = 1.0 - sharpRate
    j = 1.0 + sharpRate
    while len(sharpImages) < numberImages:
        sharpImages.append((sharpness.enhance(i), createNewName(name, i, 'sharp')))
        i = i - sharpRate
        sharpImages.append((sharpness.enhance(j), createNewName(name, j, 'sharp')))
        j = j + sharpRate
    return sharpImages

#zoomIn eliminating X pixels from every boundary.
def zoomIn(image, zoomProportion):
    croplimit1 = 0+float(image.size[0])*float(zoomProportion/2)
    croplimit2 = 0+float(image.size[1])*float(zoomProportion/2)
    croplimit3 = image.size[0]-float(image.size[0])*float(zoomProportion/2)
    croplimit4 = image.size[1]-float(image.size[1])*float(zoomProportion/2)
    croppedImage = image.crop((croplimit1, croplimit2, croplimit3, croplimit4))
    resizedImage = croppedImage.resize(image.size)
    return resizedImage

# return an array of zoomed-in images. Includes the initial image.
def multipleZoom(image, name, zoomRate, NumberOfImages):
    i = 0.00
    zoomInImages = []
    while len(zoomInImages) < NumberOfImages:
        zoomInImages.append((zoomIn(image, i), createNewName(name, i, 'zoom')))
        i = i + zoomRate
    return zoomInImages

def createNewName(baseName, rate, effect):
    #I split only the last point (associated to the extension).
    nameSplit = baseName.rsplit('.', 1)
    newName = nameSplit[0]+'-'+effect+str(rate)+'.'+nameSplit[1]
    return newName

def saveImages(imageListTuple, newDirectory):
    for im in imageListTuple:
        directory = str(newDirectory+str(im[1]))
        im[0].save(directory, 'JPEG')

''' from a file, return a list of tuples (image, name) for every image created.
(original images are included in the list also)
For the case of 80 original images (10 for each 6 and As), the function generates 8.400 images.'''
def generateSampleOfImages(realCardsDirectory):
    #first create a list of cards and a list of names, from pictures in a folder.
    realCards, names = loadImagesAndNames(realCardsDirectory)
    #realCards = rotateImages(realCards)
    realCards = scaleImages(realCards)
    #I create a list of tuples for the new images
    # and the names associated with them.
    newImages = []
    #first add zoomed-in images.
    for i, card in enumerate(realCards):
        zoomImages = multipleZoom(card, names[i], 0.05, 5)
        # here I add the new card and its type.
        newImages = newImages + zoomImages
    print(len(newImages))
    #add modified images (contrast, new color, brightness)
    modifiedList = []
    for im in newImages:
        modifiedList = modifiedList + contrast(im[0], im[1], 0.2, 2)
        modifiedList = modifiedList + color(im[0], im[1], 0.3, 1)
        modifiedList = modifiedList + brightness(im[0], im[1], 0.2, 3)
    newImages = newImages + modifiedList
    print(len(newImages))
    # I combine sharpness with all the other modifications.
    sharpList = []
    for im in newImages:
        sharpList = sharpList + sharpness(im[0], im[1], 0.5, 2)
    newImages = newImages + sharpList
    print('new set of images have '+str(len(newImages))+' images.')
    return newImages


'''To generate the cards, yo have to specify the directory where the original cards are,
and the directory where the new cards will be saved.
Do not run this before emptying the folder where the new cards will be.
'''
def main():
    realCardsDirectory = os.path.dirname(sys.argv[0]) + '/all_real_images/'
    Images = generateSampleOfImages(realCardsDirectory)
    newDirectory = os.path.dirname(sys.argv[0]) + '/generatedCards/'
    saveImages(Images, newDirectory)
    return

if __name__ == '__main__':
    main()
