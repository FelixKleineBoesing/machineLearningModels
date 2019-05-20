import numpy as np

from src.cost_functions.Cost import Cost


class LogReg(Cost):

    def compute(self, y_hat: np.ndarray, y: np.ndarray):
        """
        computes loss
        :param y_hat: y predictions
        :param y: y actuals
        :return:
        """
        return np.sum((y_hat - y) ** 2) / (2 * len(y))

    def first_order_gradient(self, y_hat: np.ndarray, y: np.ndarray, var: np.ndarray):
        """
        return first order gradient for the specified var vector
        :param y_hat:
        :param y:
        :param var:
        :return:
        """
        return np.sum(np.transpose(y_hat - y) * np.transpose(var)) / len(y)

    def second_order_gradient(self, y_hat: np.ndarray, y: np.ndarray, var: np.ndarray):
        pass


