from __future__ import division
import numpy



def activationfunction(activationFunctionName, inputHL):
    if activationFunctionName == "logSoftPlus":
        #result = numpy.log(1 + numpy.exp(inputHL))
        m = numpy.ndarray.max(inputHL)
        return m + numpy.log(numpy.exp(-m) + numpy.exp(inputHL - m))
    elif activationFunctionName == "tanh":
        return (1-numpy.exp(-2*inputHL))/(1+numpy.exp(-2*inputHL))
    elif activationFunctionName == "cosine":
        return numpy.cos(inputHL)


def softmax(inputX):
    m = numpy.ndarray.max(inputX, axis=1)
    numstab = inputX - m[:, None]  # a vector that numerically stabilized.
    x_exp = numpy.exp(numstab)
    denominator = numpy.ndarray.sum(x_exp, axis=1)
    return x_exp / denominator[:, None]


def gradderivH(actFunction, z):
    if actFunction == "logSoftPlus":
        result = (numpy.exp(z) / (numpy.exp(z) + 1))
    elif actFunction == 'tanh':
        result = 1 - activationfunction('tanh', z) ** 2
    elif actFunction == 'cosine':
        result = -numpy.sin(z)
    return result