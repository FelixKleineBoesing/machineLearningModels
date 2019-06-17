import numpy as np
import types
import pandas as pd
from typing import Union, Tuple


from src.models.Model import Model
from src.models.networks.ActivationFunctions import sigmoid, softmax, linear, relu, tanh


class NeuralNetwork(Model):
    """
    Simple neural network implementation with x dense layers, n units and some possile activationsfunctions
    """

    def __init__(self, input_shape: Tuple, neurons: list = [10, 1],
                 activation_functions: list = ["relu", "linear"], params: dict = None):
        assert isinstance(neurons, list)
        assert isinstance(activation_functions, list)
        assert isinstance(input_shape, Tuple)
        assert isinstance(input_shape, dict)
        assert len(neurons) > 0, "Number of layers must be larger than zero"
        assert len(neurons) == len(activation_functions), "Number of activation functions must be equal to length " \
                                                          "neurons list"
        assert all(isinstance(unit, int) for unit in neurons), "all number neurons must be of type integer"
        assert all([func in ["relu", "sigmoid", "tanh", "linear", "softmax"] or isinstance(func, types.FunctionType)
                    for func in activation_functions]), "Other activation functions than ReLu, sigmoid, tanh, linear " \
                                                        "or softmax are currently not supported. Supply the function " \
                                                        "itself, if you want to use another function. This function " \
                                                        "must be applicable vectorized."
        self.params = params
        self.input_shape = input_shape
        self.activation_functions = return_activation_functions(activation_functions)
        self.neurons = neurons
        super().__init__()

    def train(self, train_data: Union[pd.DataFrame, np.ndarray], train_label: Union[pd.DataFrame, np.ndarray],
              val_data: Union[pd.DataFrame, np.ndarray], val_label: Union[pd.DataFrame, np.ndarray]):
        variables = self._init_variables(train_data.shape[0])
        # TODO perform foreward pass

        # TODO backpropagate error

    def predict(self, test_data: Union[pd.DataFrame, np.ndarray]):
        pass

    def _init_variables(self, n_observations):
        variables = {}
        last_neuron = self.input_shape
        for index, neuron in enumerate(self.neurons):
            variables[index] = {
                "weights": np.random.normal(size=(last_neuron, neuron)),
                "bias": np.random.normal(size=(n_observations, ))
            }
            last_neuron = neuron
        return variables


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
            funcs.append(default_funcs[func])
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


