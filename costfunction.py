import numpy as np


def sumnorm(W1, W2):

    # Calculate the norm of the weights used in regularization
    squares1 = np.sum(np.square(W1), axis=0)
    sum1 = np.sum(squares1)
    squares2 = np.sum(np.square(W2), axis=0)
    sum2 = np.sum(squares2)
    return sum1+sum2


def costFunction(Y, T, W1, W2, reg=0.1):

    # Calculate cost function E
    E = np.sum(np.sum(np.multiply(np.log(Y), T), axis=1))-(reg/2)*sumnorm(W1, W2)

    return E
