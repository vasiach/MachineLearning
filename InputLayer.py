import numpy
from sklearn import preprocessing
from numpy import random


def getweights(X, M, K=10):  # Random initialization of weights

    weights1 = 0.1 * numpy.random.rand(M, X.shape[1])
    weights2 = 0.1 * numpy.random.rand(K, M+1)

    return weights1, weights2

def inputlayer(M, X, T):
    # M is the number of units in the hidden layer
    # X is the array containing the input extended with the bias.
    X = numpy.array(X)
    T = numpy.array(T)  # True labels of Training Examples
    W1, W2 = getweights(X, M)  # Arrays of Weights in layer 1 and 2

    print 'Dimension of arrays X and T are: ', X.shape, ' ', T.shape
    print 'Dimension of arrays W1 and W2 are: ', W1.shape, ' ', W2.shape

    # Scale all the data to values from 0-1
    # (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    X = preprocessing.minmax_scale(X, feature_range=(0, 1))
    # X_scaled = X_std * (max - min) + min
    return X, T, W1, W2
