from costfunction import costFunction
from InputLayer import inputlayer
from hidden import hiddenlayer
from output import outputlayer
from backpropagation import backpropagate
from gradientCheck import gradCheck
import numpy
import os
import re


def compareArrays(Y,T):

    if Y.shape == T.shape:
        prediction = numpy.argmax(Y, 1)
        diff = 0
        for place in range(len(T)):
            if numpy.argmax(T[place]) != prediction[place]:
                diff += 1
        percentage = float(diff)/len(prediction)
        return percentage
    else:
        print "they don't have the same shape"
        return "Invalid compare"


def load_data():
    train_path = os.path.join(os.getcwd(), "mnisttxt")
    train = {}  # Input training data
    test = {}
    labels = {}
    test_labels = {}
    for root, directories, files in os.walk(train_path):  # read the input files
        for f in files:
            name = f.replace(".txt", "")
            f1 = open(train_path + os.sep + str(f))
            data = [map(int, line.split(" ")) for line in f1.readlines()]
            for lista in range(len(data)):
                data[lista] = [float(element) / 255 for element in data[lista]]
            bias = numpy.array(numpy.ones(len(data)))
            if "train" in f:
                train[name] = numpy.column_stack((bias, data))
                array = [0 for i in range(10)]
                label = int(re.search(r'\d+', f).group())
                array[label] = 1
                labels[label] = [array for i in range(len(train[name]))]
            else:
                test[name] = numpy.column_stack((bias, data))
                t_array = [0 for i in range(10)]
                test_label = int(re.search(r'\d+', f).group())
                t_array[test_label] = 1
                test_labels[test_label] = [t_array for element in range(len(test[name]))]
            f1.close()
    temp = []
    X = []
    Xtest = []
    test_temp = []
    for i in range(10):
        temp.extend(labels[i])
        X.extend(train["train" + str(i)])
        Xtest.extend(test["test"+str(i)])
        test_temp.extend(test_labels[i])
    return X, temp, Xtest, test_temp

X, T, Xtest, labels_test = load_data()
threshold = 0.0001

labels_test = numpy.array(labels_test)
Xtest = numpy.array(Xtest)

activation_function = raw_input("Please choose one of the activation functions: 1)logSoftPlus 2)tanh 3)cosine : ")
M = input("Choose the number of activation units from : 100, 200, 300, 400, 500 : ")

gradCheckChoice=raw_input("Do you want to run gradient check?(Y/N) ")
if gradCheckChoice.lower()=="y".lower():
    gradCheck(activation_function, M, X, T)
    
error = 0
error_prev = -numpy.inf
X, T, W1, W2 = inputlayer(M, X, T)
n = 0.5/X.shape[0]
iter = 1000
for epoch in range(iter):
    print "-------------epoch %s-------------" %epoch
    Z1, A2 = hiddenlayer(X, W1, activation_function)
    Y, Z2 = outputlayer(A2, W2)
    E = costFunction(Y, T, W1, W2)
    error = E
    print 'Error returned: ', E
    UpW2, UpW1 = backpropagate(X, Y, T, Z1, A2,  W1, W2, activation_function)
    if numpy.absolute(error - error_prev) < threshold:
        break
    W1 += n*UpW1
    W2 += n*UpW2
    print "the difference is",numpy.absolute(error - error_prev)
    error_prev = E

#after the iteration we will have the best possible weights for our neural network W1 and W2
Z1, A2 = hiddenlayer(Xtest, W1, activation_function)
Y, Z2 = outputlayer(A2, W2)
#true labels of the tests is labels_tests

print "they differ by %s per cent" % (compareArrays(Y, labels_test))
