from sklearn.linear_model import SGDClassifier
import numpy as np
import random
from load_partial_dataset import loadImages
import math

'''
Just checking how other standard models perform in this dataset.
Multiclass Logistic Regression (with OVA)
Multiclass Support Vector Machines (with OVA)
'''
random.seed(1)
train = [random.randint(0,52000) for i in range(2000)]
dev = [random.randint(0,52000) for i in range(500)]
#dev = list(range(0, 52000, 100))
dev = [x for x in dev if x not in train]

'''
load X, Y vectors for train and dev.
'''
X,Y = loadImages(train, 'vectorY52.npy')
X_flattened = X.reshape((len(X),120*160*3))
del X
# Convert vectors to value
Y_values = [list(y).index(1) for y in Y]
del Y
X_dev, Y_dev = loadImages(dev, 'vectorY52.npy')
X_dev = X_dev.reshape((len(X_dev),120*160*3))
Y_dev = [list(y).index(1) for y in Y_dev]

'''
lambda used in each of the models.
'''
alpha = 0.1
#alpha = 0.02


'''
Model performance by iteration.
'''
def modelsByIteration(X_flattened, Y_values, X_dev, Y_dev, modelType, alpha):
	iterations = 2
	while iterations <= 30:
		model = SGDClassifier(loss = modelType, alpha = alpha, max_iter = iterations)
		#model = SGDClassifier(loss = 'hinge', penalty = 'none', max_iter = iterations)
		model.fit(X_flattened, Y_values)
		print iterations
		print 'iteration = '+str(iterations)
		print('Training Accuracy:')
		print(model.score(X_flattened,Y_values))
		model.predict(X_dev)
		print('Dev Accuracy:')
		print(model.score(X_dev,Y_dev))
		iterations = iterations + 2

'''
Model performance for different lambdas.
'''
def chooseLambda(X_flattened, Y_values, X_dev, Y_dev, modelType):
	learnrate = -4.0
	accuraciesT = []
	accuraciesD = []
	rates = [x for x in range(0, 11)]
	lambdas = []

	for rate in rates:
		alpha = math.pow(10, (learnrate*rate/12))
		lambdas.append(alpha)
		print 'lambda', alpha
		model = SGDClassifier(loss = modelType, alpha = alpha, max_iter =30)
		model.fit(X_flattened, Y_values)
		print('Training Accuracy:')
		accuracyT = model.score(X_flattened,Y_values)
		print(accuracyT)
		accuraciesT.append(float(accuracyT))
		model.predict(X_dev)
		print('Dev Accuracy:')
		accuracyD = model.score(X_dev,Y_dev)
		print(accuracyD)
		accuraciesD.append(float(accuracyD))


#chooseLambda(X_flattened, Y_values, X_dev, Y_dev, 'log')
modelsByIteration(X_flattened, Y_values, X_dev, Y_dev, 'log', 0.1)
