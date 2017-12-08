from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from load_partial_dataset import loadImages
from load_new_dataset import loadImagesX
import random

'''
For training the model in the original dataset, but testing it 
with the new images from the web.
Here you can train each of the models.
'''
def main():

    train = list(range(0, 8400, 13))
    arr = np.load('vectorY3.npy')
    X,Y = loadImages(train, arr)
    X_flattened = X.reshape((len(X),480*640*3))
    del X
    # Convert vectors to value
    Y_values = [list(y).index(1) for y in Y]
    del Y
    #model = linear_model.LogisticRegression(solver='newton-cg', multi_class='multinomial')
    model = SVC(kernel='linear', C=1)
    #model = LinearDiscriminantAnalysis()
    #model = GaussianNB()

    model.fit(X_flattened,Y_values)
    del X_flattened,Y_values

    # 100 testing sets of 200 images each.

    devSets = []
    devSets.append(range(0,55))
    #for _ in range(2):
        #devSets.append(random.sample(range(0, 8400), 200))

    results = []
    arrX = np.load('vectorY3X.npy')
    for dev in devSets:
        X_dev, Y_dev = loadImagesX(dev, arrX)
        X_dev = X_dev.reshape((len(X_dev),480*640*3))
        Y_dev = [list(y).index(1) for y in Y_dev]
        p = model.predict(X_dev)
        result = model.score(X_dev,Y_dev)
        results.append(result)

    print 'Average Accuracy: '+str(np.mean(results))


if __name__ == '__main__':
    main()
