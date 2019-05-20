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
