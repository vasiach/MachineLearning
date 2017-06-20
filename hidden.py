import numpy
from tools import activationfunction


def hiddenlayer(X, W1, function_name):

    # Calculate Z1 the array containing the input to the hidden unit
    Z1 = numpy.dot(X, W1.T)

    # Call activation function
    H = activationfunction(function_name, Z1)

    # Add bias unit on hidden layer
    A2 = numpy.concatenate((numpy.ones((H.shape[0], 1)), H), axis=1)

    return Z1, A2
