import numpy as np

from src.cost_functions.Cost import Cost
from src.cost_functions.Helper import reshape_outputs


class MeanSqaredError(Cost):

    def compute(self, y_hat: np.ndarray, y: np.ndarray, aggregation: bool = False):
        """
        computes loss
        :param y_hat: y predictions
        :param y: y actuals
        :return:
        """
        y_hat, y = reshape_outputs(y_hat, y)
        cost = np.power(y_hat - y, 2) / 2
        return np.sum(cost) / len(y)

    def first_order_gradient(self, y_hat: np.ndarray, y: np.ndarray):
        """
        return first order gradient for the specified var vector
        :param y_hat:
        :param y:
        :param var:
        :return:
        """
        y_hat, y = reshape_outputs(y_hat, y)
        return np.transpose(y_hat - y)

    def second_order_gradient(self, y_hat: np.ndarray, y: np.ndarray):
        pass




