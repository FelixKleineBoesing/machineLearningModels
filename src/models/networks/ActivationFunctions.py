import numpy as np


def sigmoid(x: np.ndarray):
    """
    sigmoid
    :param x: input data
    :return:
    """
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray):
    """
    tanh
    :param x:
    :return:
    """
    return 1 - (2 / (np.exp(2*x) - 1))


def relu(x: np.ndarray):
    """
    simply forwards positive values and eliminates negative values
    :param x:
    :return:
    """
    x[x < 0] = 0
    return x


def linear(x: np.ndarray):
    """
    simply forwards input value
    :param x:
    :return:
    """
    return x


def softmax(x: np.ndarray):
    """
    all values some to one
    :param x:
    :return:
    """
    assert(len(x.shape) == 2), "x must be two dimensional"
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

