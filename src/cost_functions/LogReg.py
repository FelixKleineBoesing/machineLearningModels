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
        cost = - y * np.log(y_hat) - (1-y) * np.log(1-y_hat)
        return np.nansum(cost[np.isfinite(cost)]) / len(y)

    def first_order_gradient(self, y_hat: np.ndarray, y: np.ndarray, var: np.ndarray):
        """
        return first order gradient for the specified var vector
        :param y_hat:
        :param y:
        :param var:
        :return:
        """
        grad = np.transpose(y_hat-y) @ var
        return np.sum(grad[np.isfinite(grad)]) / len(y)

    def second_order_gradient(self, y_hat: np.ndarray, y: np.ndarray, var: np.ndarray):
        pass
