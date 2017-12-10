from sklearn import linear_model
import numpy as np
import random
from load_partial_dataset import loadImages

def main():
    random.seed(1)
    random_train = [random.randint(0,8400) for i in range(100)]
    random_dev = [random.randint(0,8400) for i in range(500)]

    train = list(range(0, 8400, 50))
    dev = list(range(0, 8400, 49))

    arr = np.load('vectorY52.npy')

    X,Y = loadImages(random_train, arr)
    X_flattened = X.reshape((len(X),120*160*3))

    del X
    # Convert vectors to value
    Y_values = [list(y).index(1) for y in Y]
    #Y_values = random.shuffle(Y_values)
    Y_values = np.asarray(Y_values)
    #Y_values = Y_values.reshape(Y_values.shape[0], -1)
    del Y

    model = linear_model.LogisticRegression(solver='newton-cg',multi_class='multinomial')
    model.fit(X_flattened, Y_values)

    print('Training Accuracy:')
    print(model.score(X_flattened,Y_values))
    p1 = model.predict(X_flattened)
    p1 = np.asarray(p1)
    print(p1)
    print(Y_values)

    del X_flattened, Y_values

<<<<<<< HEAD
    X_dev, Y_dev = loadImages(random_dev, arr)
    X_dev = X_dev.reshape((len(X_dev),120*160*3))
=======
    X_dev, Y_dev = loadImages(dev, arr)
    X_dev = X_dev.reshape((len(X_dev),480*640*3))
>>>>>>> 51a2461024995989530b2a09b509d35af6a32637
    Y_dev = [list(y).index(1) for y in Y_dev]
    p = model.predict(X_dev)
    p = np.asarray(p)
    
    print('Accuracy:')
    print(model.score(X_dev,Y_dev))

if __name__ == '__main__':
    main()
