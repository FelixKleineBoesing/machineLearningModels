import numpy as np
import abc


class ActivationFunction(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def compute(self, x: np.ndarray):
        pass

    @abc.abstractmethod
    def first_order_gradient(self, x: np.ndarray):
        pass

    @abc.abstractmethod
    def second_order_gradient(self, x: np.ndarray):
        pass


class sigmoid(ActivationFunction):

    def compute(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def first_order_gradient(self, x: np.ndarray):
        return self.compute(x) * (1 - self.compute(x))

    def second_order_gradient(self, x: np.ndarray):
        pass


class tanh(ActivationFunction):

    def compute(self, x: np.ndarray):
        return 1 - (2 / (np.exp(2 * x) - 1))

    def first_order_gradient(self, x: np.ndarray):
        return 1 - self.compute(x) ** 2

    def second_order_gradient(self, x: np.ndarray):
        pass


class relu(ActivationFunction):

    def compute(self, x: np.ndarray):
        x[x < 0] = 0
        return x

    def first_order_gradient(self, x: np.ndarray):
        x[x < 0] = 0
        x[x > 0] = 1
        return x

    def second_order_gradient(self, x: np.ndarray):
        pass


class linear(ActivationFunction):

    def compute(self, x: np.ndarray):
        return x

    def first_order_gradient(self, x: np.ndarray):
        return np.ones(x.shape)

    def second_order_gradient(self, x: np.ndarray):
        pass


class softmax(ActivationFunction):

    def compute(self, x: np.ndarray):
        assert (len(x.shape) == 2), "x must be two dimensional"
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def first_order_gradient(self, x: np.ndarray):
        return self.compute(x) * (1 - self.compute(x))

    def second_order_gradient(self, x: np.ndarray):
        pass
