import numpy as np
from tools import gradderivH


def error_output(Y, T):
    # delta error on output layer
    delta_output = T-Y
    return delta_output


def out_grad_weights_w2(W2, d_out, A2, reg=0.1):

    # Update rule for W2 weights
    UpdW2 = np.dot(d_out.T, A2) - reg*W2

    return UpdW2

def out_grad_weights_w1(W2, d_out, W1, X, Z1, actFunction, reg=0.1):
    # Calculate derivative of activation function
    grad_actfunction = gradderivH(actFunction, Z1)

    #Add bias
    #grad_actfunction = np.c_[np.ones((grad_actfunction.shape[0], 1)), grad_actfunction]
    
    #delta error on hidden layer
    d_1 = np.dot(W2[:, 1:].T, d_out.T) * grad_actfunction.T

    #Update rule for W1 weights
    UpdW1 = np.dot(d_1, X) - reg*W1
    return UpdW1


def backpropagate(X, Y, T, Z1, A2,  W1, W2, actFunction):
    d_out = error_output(Y, T)
    UpdW2 = out_grad_weights_w2(W2, d_out, A2)
    UpdW1 = out_grad_weights_w1(W2, d_out, W1, X, Z1, actFunction)
    return UpdW2, UpdW1

