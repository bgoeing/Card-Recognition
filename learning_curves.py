from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from load_partial_dataset import loadImages
import random


'''
In this file I tested the models with different training set sizes.

'''

def testModel(model):

    train0 = list(range(0, 8400, 332)) #25 images
    train1 = list(range(0, 8400, 170)) #50 images
    train2 = list(range(0, 8400, 85)) #100 images
    train3 = list(range(0, 8400, 58)) #150 images
    train4 = list(range(0, 8400, 44)) #200 images
    train5 = list(range(0, 8400, 32)) #250 images
    train6 = list(range(0, 8400, 23)) #400 images
    train7 = list(range(0, 8400, 12)) #600 images
    arr = np.load('vectorY3.npy')
    X,Y = loadImages(train3, arr)
    X_flattened = X.reshape((len(X),480*640*3))
    del X
    # Convert vectors to value
    Y_values = [list(y).index(1) for y in Y]
    del Y
    model.fit(X_flattened,Y_values)
    devSets = []
    for _ in range(10):
        devSets.append(random.sample(range(0, 8400), 200))
    results = []
    for dev in devSets:
        X_dev, Y_dev = loadImages(dev, arr)
        X_dev = X_dev.reshape((len(X_dev),480*640*3))
        Y_dev = [list(y).index(1) for y in Y_dev]
        p = model.predict(X_dev)
        result = model.score(X_dev,Y_dev)
        results.append(result)
        print result

    print 'Average Accuracy: '+str(np.mean(results))

#model = SVC(kernel='linear', C=1)
#model = LinearDiscriminantAnalysis()
#model = GaussianNB()
model = LogisticRegression(solver='newton-cg',multi_class='multinomial')

testModel(model)


