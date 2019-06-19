import numpy as np
import types
import logging
import pandas as pd
from typing import Union, Tuple


from src.models.Model import Model
from src.models.networks.NetworkHelper import return_activation_functions
from src.models.networks.ActivationFunctions import ActivationFunction
from src.cost_functions.Cost import Cost


class NeuralNetwork(Model):
    """
    Simple neural network implementation with x dense layers, n units and some possile activationsfunctions
    """

    def __init__(self, cost_function: Cost, input_shape: Tuple, neurons: list = [10, 1], params: dict = None,
                 activation_functions: list = ["relu", "linear"], epochs: int = 10, learning_rate: float = 0.1):
        assert isinstance(cost_function, Cost)
        assert isinstance(neurons, list)
        assert isinstance(activation_functions, list)
        assert isinstance(input_shape, Tuple)
        assert isinstance(params, dict)
        assert len(neurons) > 0, "Number of layers must be larger than zero"
        assert len(neurons) == len(activation_functions), "Number of activation functions must be equal to length " \
                                                          "neurons list"
        assert all(isinstance(unit, int) for unit in neurons), "all number neurons must be of type integer"
        assert all([func in ["relu", "sigmoid", "tanh", "linear", "softmax"] or isinstance(func, ActivationFunction)
                    for func in activation_functions]), "Other activation functions than ReLu, sigmoid, tanh, linear " \
                                                        "or softmax are currently not supported. Supply the function " \
                                                        "itself, if you want to use another function. This function " \
                                                        "must be applicable vectorized."

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.params = params
        self.input_shape = input_shape
        self.activation_functions = return_activation_functions(activation_functions)
        self.neurons = neurons
        self.cost_function = cost_function
        self.network = self._init_variables()
        super().__init__()

    def train(self, train_data: Union[pd.DataFrame, np.ndarray], train_label: Union[pd.DataFrame, np.ndarray],
              val_data: Union[pd.DataFrame, np.ndarray], val_label: Union[pd.DataFrame, np.ndarray]):
        stopped = False
        epoch = 1
        while not stopped and epoch < self.epochs:
            y_hat = self._forward_pass(train_data)
            costs = self.cost_function.compute(y_hat, train_label)
            self._backward_pass(y_hat, train_label)
            print("Epoch {}, train loss: {}".format(epoch, costs))
            epoch += 1

    def _forward_pass(self, data: np.ndarray):
        arr = data
        for key, layer in self.network.items():
            arr = layer["activ"].compute(arr @ layer["weights"] + layer["bias"])
        return arr

    def _backward_pass(self, y_hat: np.ndarray, y: np.ndarray):
        cost_gradient = self.cost_function.first_order_gradient(y_hat, y)
        for key in reversed(list(self.network.keys())):
            layer = self.network[key]
            activ_gradient = layer["activ"].first_order_gradient(cost_gradient)
            if len(np.array(activ_gradient).shape) == 0:
                activ_gradient = np.array([[activ_gradient], ])
            if len(activ_gradient.shape) == 1:
                activ_gradient = activ_gradient.reshape(activ_gradient.shape[0], 1)
            cost_gradient = (layer["weights"] @ activ_gradient)
            layer["weight"] = layer["weights"] - self.learning_rate * cost_gradient
            self.network[key] = layer


    def predict(self, test_data: Union[pd.DataFrame, np.ndarray]):
        pass

    def _init_variables(self):
        variables = {}
        last_neuron = self.input_shape[0]
        for index, neuron in enumerate(self.neurons):
            variables[index] = {
                "weights": np.random.normal(size=(last_neuron, neuron)),
                "bias": np.random.normal(size=(1, neuron)),
                "activ": self.activation_functions[index]
            }
            last_neuron = neuron
        return variables
