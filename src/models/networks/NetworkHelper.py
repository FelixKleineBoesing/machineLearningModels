import types

import numpy as np

from src.models.networks.ActivationFunctions import relu, tanh, sigmoid, linear, softmax


def return_activation_functions(activ_funcs: list):
    """
    maps strings to functions. If function delivered in list it will be checked whether it outputs the right shape
    :param activ_funcs: list of activation functions (str or functions)
    :return: a list of functions
    """
    default_funcs = {"relu": relu, "tanh": tanh, "sigmoid": sigmoid, "linear": linear, "softmax": softmax}
    funcs = []
    for func in activ_funcs:
        if isinstance(func, str):
            funcs.append(default_funcs[func]())
        else:
            _check_validity(func)
            funcs.append(func)

    return funcs


def _check_validity(func: types.FunctionType):
    test_data1 = np.array([1, 2, 3])
    test_data2 = np.array([[1, 2, 3]])
    test_data3 = np.array([[[1, 2, 3]]])

    result_data1 = func(test_data1)
    result_data2 = func(test_data2)
    result_data3 = func(test_data3)

    assert result_data1.shape == test_data1.shape, "Function {} is not valide, because shape differs from input to " \
                                                   "output. Inputshape: {}, Outputshape: {}". \
        format(func.__name__, test_data1.shape, result_data1.shape)
    assert result_data1.shape == test_data1.shape, "Function {} is not valide, because shape differs from input to " \
                                                   "output. Inputshape: {}, Outputshape: {}". \
        format(func.__name__, test_data2.shape, result_data2.shape)
    assert result_data2.shape == test_data2.shape, "Function {} is not valide, because shape differs from input to " \
                                                   "output. Inputshape: {}, Outputshape: {}". \
        format(func.__name__, test_data3.shape, result_data3.shape)