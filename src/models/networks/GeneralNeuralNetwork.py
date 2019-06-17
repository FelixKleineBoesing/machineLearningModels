import numpy as np
from typing import Union
import pandas as pd
from src.models.Model import Model


class NeuralNetwork(Model):

    def __init__(self, hidden_layers: int = 1, neurons: list = None, activation_functions: list = None,
                 params: dict = None):
        assert hidden_layers > 0, "Hidden layers must be larger than one"
        assert hidden_layers == len(neurons), "Number of hidden layers must be equal to length neurons list"
        assert hidden_layers == len(activation_functions), "Number of activation functions must be equal to length " \
                                                           "neurons list"
        self.params = params
        super().__init__()


    def train(self, train_data: Union[pd.DataFrame, np.ndarray], train_label: Union[pd.DataFrame, np.ndarray],
              val_data: Union[pd.DataFrame, np.ndarray], val_label: Union[pd.DataFrame, np.ndarray]):
        pass

    def predict(self, test_data: Union[pd.DataFrame, np.ndarray]):
        pass





class Layer:

    def __init__(self, units: int, activation_function: str = "ReLu"):
        assert isinstance(units, int)
        assert isinstance(activation_function, str)
        assert activation_function in ["ReLu", "sigmoid", "tanh", "linear", "softmax"], \
            "Other activation functions than ReLu, sigmoid, tanh, linear or softmax are currently not supported. " \
            "Supply the function itself, if you want to use another function. This function must be applicable " \
            "vectorized."

        self.units = units
        self.activation_function = activation_function

        pass