import numpy as np
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

    def __init__(self, cost_function: Cost, input_shape: Tuple, neurons: list = [4, 8, 1], params: dict = None,
                 activation_functions: list = ["relu", "relu", "linear"], epochs: int = 10, learning_rate: float = 0.1,
                 verbose: bool = False):
        assert isinstance(neurons, list)
        assert isinstance(activation_functions, list)
        assert isinstance(input_shape, Tuple)
        assert isinstance(params, dict)
        assert isinstance(verbose, bool)
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
        self.verbose = verbose
        self.network = self._init_variables()
        super().__init__()

    def train(self, train_data: Union[pd.DataFrame, np.ndarray], train_label: Union[pd.DataFrame, np.ndarray],
              val_data: Union[pd.DataFrame, np.ndarray], val_label: Union[pd.DataFrame, np.ndarray]):
        assert (val_label is not None and val_label is not None) or (val_label is None and val_data is None)
        stopped = False
        epoch = 1
        while not stopped and epoch < self.epochs:
            y_hat = self._forward_pass(train_data)
            if val_data is not None:
                y_hat_val = self.predict(val_data)
                val_costs = self.cost_function.compute(y_hat_val, val_label, aggregation=True)
            costs = self.cost_function.compute(y_hat, train_label, aggregation=True)
            self._backward_pass(y_hat, train_label, train_data)
            if self.verbose:
                if val_data is not None:
                    print("Epoch {}, train loss: {}, val_loss: {}".format(epoch, costs, val_costs))
                else:
                    print("Epoch {}, train loss: {}".format(epoch, costs))

            epoch += 1

    def _forward_pass(self, data: np.ndarray):
        arr = data
        for key, layer in self.network.items():
            activation = arr @ layer["weights"] + layer["bias"]
            arr = layer["activ"].compute(activation.copy())
            layer["activation"] = activation
        return arr

    def _backward_pass(self, y_hat: np.ndarray, y: np.ndarray, X: np.ndarray):
        weight_updates = {}
        past_layer = None
        for index, key in enumerate(reversed(list(self.network.keys()))):
            layer = self.network[key]
            if index > 0:
                weight_updates[past_layer_key] = self._clipp_gradients(layer["activation"].T @ activ_gradient)
            if index == 0:
                cost_gradient = self.cost_function.first_order_gradient(y_hat, y).T
            else:
                cost_gradient = (activ_gradient @ past_layer["weights"].T)
            activ_gradient = cost_gradient * layer["activ"].first_order_gradient(layer["activation"])
            activ_gradient = _reshape_activ_gradient(activ_gradient)
            past_layer = layer
            past_layer_key = key

        weight_updates[key] = X.T @ activ_gradient

        for key, wu in weight_updates.items():
            self.network[key]["weights"] = self.network[key]["weights"] - self.learning_rate * wu

    def predict(self, test_data: Union[pd.DataFrame, np.ndarray]):
        arr = test_data
        for key, layer in self.network.items():
            activation = arr @ layer["weights"] + layer["bias"]
            arr = layer["activ"].compute(activation)
        return arr

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

    def _clipp_gradients(self, arr):
        arr[arr > 100] = 100
        arr[arr < -100] = -100
        return arr


def _reshape_activ_gradient(grad: np.ndarray):
    if len(np.array(grad).shape) == 0:
        grad = np.array([[grad], ])
    if len(grad.shape) == 1:
        grad = grad.reshape(grad.shape[0], 1)
    return grad