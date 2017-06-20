import numpy as np
from costfunction import costFunction
from InputLayer import inputlayer
from hidden import hiddenlayer
from output import outputlayer
from backpropagation import backpropagate

def gradCheck(activation_function, M, X, T):
    
    X, T, W1, W2 = inputlayer(M, X, T)
    
    #We take a sample of X and T
    listSample = np.random.randint(X.shape[0], size=30)
    XSample = np.array(X[listSample, :])
    TSample = np.array(T[listSample, :])
    Z1, A2 = hiddenlayer(XSample, W1, activation_function)
    Y, Z2 = outputlayer(A2, W2)
    UpW2, UpW1 = backpropagate(XSample, Y, TSample, Z1, A2,  W1, W2, activation_function)
    
    gradW1 = np.zeros(UpW1.shape)
    gradW2 = np.zeros(UpW2.shape)
    
    epsilon=1e-4
    # gradcheck for w1
    for i in range(0, gradW1.shape[0]):
        for j in range(0, gradW1.shape[1]):
            tmpW = np.copy(W1)
            tmpW[i,j] += epsilon
            EPlus = costFunction(Y, TSample, tmpW, W2)

            tmpW = np.copy(W1)
            tmpW[i,j] -= epsilon
            EMinus = costFunction(Y, TSample, tmpW, W2)
            gradW1[i,j] = (EPlus - EMinus) / (2 * epsilon)
            
    print "The difference estimate for gradient of W1 is: ", np.amax(np.abs(UpW1 - gradW1))
    
    # gradcheck for w2
    for i in range(0, gradW2.shape[0]):
        for j in range(0, gradW2.shape[1]):
            tmpW = np.copy(W2)
            tmpW[i,j] += epsilon
            EPlus = costFunction(Y, TSample, W1, tmpW)

            tmpW = np.copy(W2)
            tmpW[i,j] -= epsilon
            EMinus = costFunction(Y, TSample, W1, tmpW)
            gradW2[i,j] = (EPlus - EMinus) / (2 * epsilon)
            
    print "The difference estimate for gradient of W2 is: ", np.amax(np.abs(UpW2 - gradW2))