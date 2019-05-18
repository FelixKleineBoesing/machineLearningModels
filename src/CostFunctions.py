import numpy as np
import abc


class Cost(abc.ABC):

    @abc.abstractmethod
    def compute(self, y_hat: np.ndarray, y: np.ndarray):
        pass

    @abc.abstractmethod
    def first_order_gradient(self, y_hat: np.ndarray, y: np.ndarray, var: np.ndarray):
        pass

    @abc.abstractmethod
    def second_order_gradient(self, y_hat: np.ndarray, y: np.ndarray, var: np.ndarray):
        pass


class MeanSqaredError(Cost):

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
        return np.sum(np.transpose((y_hat - y) @ var)) / len(y)

    def second_order_gradient(self, y_hat: np.ndarray, y: np.ndarray, var: np.ndarray):
        pass
