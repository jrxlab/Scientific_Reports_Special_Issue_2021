import numpy as np_


def Sigmoid(min_, max_, len_, height):
    t = np_.linspace(min_, max_, len_)
    y = height / (1.0 + np_.exp(-t))

    return y


def Sigmoid_Inv(min_, max_, len_, height):

    sigmoid = Sigmoid(min_, max_, len_, height)

    return 1.0 - sigmoid
