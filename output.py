import numpy as np
from tools import softmax



def outputlayer(A2, W2):

    # Calculate Z2 which is the array containing the input of the output layer
    Z2 = np.dot(A2, W2.T)

    # Calculate Y which is the output of the output unit after activating softmax on Z2
    Y = softmax(Z2)

    return Y, Z2
