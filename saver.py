import pickle
import os,sys

directory = os.path.dirname(sys.argv[0]) + '/savedParameters/'

def loadParameters(name):
    with open(directory + name, 'rb') as f:
        return pickle.load(f)

def saveParameters(params,name):
    with open(directory + name, 'wb') as f:
        pickle.dump(params, f)