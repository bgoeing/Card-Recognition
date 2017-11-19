from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from load_partial_dataset import loadImages

'''
Just checking how other standard models perform in this datasets.
The model takes maybe 5-10 minutes to run, so I plugged here the results:

Logistic Regression 0.994186046512
Linear Discriminant Analysis 0.895348837209
K-means 0.424418604651
Gaussian Naive Bayes 0.447674418605
Support Vector Machines 0.53488372093
'''

train = list(range(0, 8400, 50))
dev = list(range(0, 8400, 49))

arr = np.load('vectorY1.npy')
X,Y = loadImages(train, arr)
X_flattened = X.reshape((len(X),480*640*3))
del X
# Convert vectors to value
Y_values = [list(y).index(1) for y in Y]
del Y

X_dev, Y_dev = loadImages(dev, arr)
X_dev = X_dev.reshape((len(X_dev),480*640*3))
Y_dev = [list(y).index(1) for y in Y_dev]

models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('K-means', KNeighborsClassifier()))
models.append(('Gaussian Naive Bayes', GaussianNB()))
models.append(('Support Vector Machines', SVC()))

results = []
names = []
for name, model in models:
	model.fit(X_flattened, Y_values)
	model.predict(X_dev)
	results.append(model.score(X_dev,Y_dev))
	names.append(name)


for i, result in enumerate(results):
	print names[i], result

