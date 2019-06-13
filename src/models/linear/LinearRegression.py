import numpy as np
import pandas as pd
from typing import Union

from src.cost_functions.MeanSquaredError import MeanSqaredError
from src.cost_functions.Cost import Cost
from src.models.Model import Model


class LinearRegression(Model):

    def __init__(self, alpha: float = 0.2, iterations: int = 10, error_function: Union[str, Cost] = "mse",
                 verbose: bool = False):
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
        self.verbose = verbose

        if error_function == "mse":
            self.error_function = MeanSqaredError()
        else:
            raise NotImplementedError("Other error functions than mse are currently not implemented")

        self._theta = None
        super().__init__()

    def train(self, train_data: Union[pd.DataFrame, np.ndarray], train_label: Union[pd.DataFrame, np.ndarray, list],
              val_data: Union[pd.DataFrame, np.ndarray]=None, val_label: Union[pd.DataFrame, np.ndarray]=None):
        """
        trains linear regression
        :param train_data:
        :param train_label:
        :param val_data:
        :param val_label:
        :return:
        """
        assert((val_data is not None and val_label is not None) or (val_label is None and val_data is None))
        assert (type(train_data) in [pd.DataFrame, np.ndarray])
        assert (type(train_label) in [pd.DataFrame, np.ndarray, list])
        if val_data is not None:
            assert (type(val_data) in [pd.DataFrame, np.ndarray])
            assert (type(val_label) in [pd.DataFrame, np.ndarray, list])
        if isinstance(train_data, pd.DataFrame):
            train_data = np.array(train_data.values)
        if isinstance(train_label, pd.DataFrame):
            train_label = np.array(train_label.values)
        elif isinstance(train_label, list):
            train_label = np.array(train_label)
        if val_data is not None:
            if isinstance(val_data, pd.DataFrame):
                val_data = np.array(val_data.values)
            if isinstance(val_label, pd.DataFrame):
                train_label = np.array(val_label.values)
            elif isinstance(val_label, list):
                train_label = np.array(val_label)

        # add bias
        train_data = np.concatenate([np.ones((train_data.shape[0], 1)), train_data], axis=1)
        if val_data is not None:
            val_data = np.concatenate([np.ones((val_data.shape[0], 1)), val_data], axis=1)
            if val_label.ndim == 1:
                val_label = val_label.reshape((val_label.shape[0], 1))
        self._theta = np.random.rand(train_data.shape[1]).reshape((train_data.shape[1], 1))

        # reshape label if necessary
        if train_label.ndim == 1:
            train_label = train_label.reshape((train_label.shape[0], 1))

        for i in range(self.iterations):
            temp_theta = np.zeros((self._theta.shape[0], 1))
            y_hat = train_data @ self._theta
            for j in range(self._theta.shape[0]):
                temp_theta[j] = self._theta[j] - self.alpha * \
                                self.error_function.first_order_gradient(y_hat, train_label, train_data[:, j].
                                                                          reshape(train_data.shape[0], 1))
            self._theta = temp_theta

            if self.verbose:
                if val_data is None:
                    print("Train loss: {}".format(self.error_function.compute(train_data @ self._theta, train_label)))
                else:
                    print("Train loss: {},  Val loss: {}".format(
                        self.error_function.compute(train_data @ self._theta, train_label),
                        self.error_function.compute(val_data @ self._theta, val_label)))

    def predict(self, test_data: Union[pd.DataFrame, np.ndarray]):
        """

        :param test_data:
        :return:
        """
        assert (self._theta is not None), "Model isn´t trained yet. Train first!"
        assert (type(test_data) in [pd.DataFrame, np.ndarray])
        if isinstance(test_data, pd.DataFrame):
            test_data = np.array(test_data.values)
        test_data = np.concatenate([np.ones((test_data.shape[0], 1)), test_data], axis=1)

        return test_data @ self._theta
