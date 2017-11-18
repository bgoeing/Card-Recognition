from sklearn import linear_model
import numpy as np
from load_partial_dataset import loadImages
import random

def main():
    random_train = [random.randint(0,8400) for i in range(300)]
    random_dev = [random.randint(0,8400) for i in range(100)]

    arr = np.load('vectorY1.npy')
    X,Y = loadImages(random_train, arr)
    X_flattened = X.reshape((len(X),480*640*3))
    del X
    # Convert vectors to value
    Y_values = [list(y).index(1) for y in Y]
    del Y
    model = linear_model.LogisticRegression(solver='newton-cg',multi_class='multinomial')
    model.fit(X_flattened,Y_values)
    del X_flattened,Y_values

    X_dev, Y_dev = loadImages(random_dev, arr)
    X_dev = X_dev.reshape((len(X_dev),480*640*3))
    Y_dev = [list(y).index(1) for y in Y_dev]
    p = model.predict(X_dev)
    
    print('Accuracy:')
    print(model.score(X_dev,Y_dev))

if __name__ == '__main__':
    main()