import numpy as np
import pandas as pd
from typing import Union
import abc

from src.CostFunctions import MeanSqaredError
from src import Model


class LinearRegression(Model):

    def __init__(self, alpha: float = 0.2, iterations: int = 10, error_function: str = "mse"):
        """

        :param alpha: learning rate
        :param iterations:  number or iterations
        """
        assert (type(error_function) == str)
        assert (type(iterations) == int)
        assert (type(alpha) == float)
        assert (iterations > 1)
        assert (alpha > 0)
        self.iterations = iterations
        self.alpha = alpha
        self._theta = None
        if error_function == "mse":
            self._error_function = MeanSqaredError()
        else:
            raise NotImplementedError("Other error functions than mse are currently not implemented")

    def train(self, train_data: Union[pd.DataFrame, np.ndarray], train_label: Union[pd.DataFrame, np.ndarray, list],
              val_data: Union[pd.DataFrame, np.ndarray], val_label: Union[pd.DataFrame, np.ndarray]):
        """
        trains linear regression
        :param train_data:
        :param train_label:
        :param val_data:
        :param val_label:
        :return:
        """
        assert (train_data in [pd.DataFrame, np.ndarray])
        assert (train_label in [pd.DataFrame, np.ndarray, list])
        assert (val_data in [pd.DataFrame, np.ndarray])
        assert (val_label in [pd.DataFrame, np.ndarray, list])
        if isinstance(train_data, pd.DataFrame):
            train_data = np.array(train_data.values)
        if isinstance(train_label, pd.DataFrame):
            train_label = np.array(train_label.values)
        elif isinstance(train_label, list):
            train_label = np.array(train_label)
        if isinstance(val_data, pd.DataFrame):
            val_data = np.array(val_data.values)
        if isinstance(val_label, pd.DataFrame):
            train_label = np.array(val_label.values)
        elif isinstance(val_label, list):
            train_label = np.array(val_label)

        # add bias
        train_data = np.concatenate(np.ones((train_data.shape[0], 1)), train_data, axis=1)
        self._theta = np.random.rand(train_data.shape[1])

        for i in range(self.iterations):
            temp_theta = np.ones((self._theta.shape[0], 1))
            y_hat = train_data @ self._theta
            for j in range(self._theta.shape[0]):
                temp_theta[j] = self._theta[j] - self._error_function.first_order_gradient(y_hat, train_label, train_data[:, j])
            self._theta = temp_theta

    def predict(self, test_data: Union[pd.DataFrame, np.ndarray]):
        """

        :param test_data:
        :return:
        """
        assert (self._theta is not None), "Model isn´t trained yet. Train first!"
        assert (test_data in [pd.DataFrame, np.ndarray])
        if isinstance(test_data, pd.DataFrame):
            test_data = np.array(test_data.values)
    




