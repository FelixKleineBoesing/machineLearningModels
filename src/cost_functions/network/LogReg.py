import numpy as np

from src.cost_functions.Cost import Cost
from src.cost_functions.Helper import reshape_outputs


class LogReg(Cost):

    def compute(self, y_hat: np.ndarray, y: np.ndarray, aggregation: bool = False):
        """
        computes loss
        :param y_hat: y predictions
        :param y: y actuals
        :return:
        """
        y_hat, y = reshape_outputs(y_hat, y)
        # pretty dirty code to prevent having 0 in np.log, since log can´t be evaluated at value 0
        y_hat[y_hat == 1.0] = 0.99999
        y_hat[y_hat == 0.0] = 0.00001
        cost = - y * np.log(y_hat) - (1-y) * np.log(1-y_hat)
        if aggregation:
            cost = np.nansum(cost[np.isfinite(cost)]) / len(y)

        return cost

    def first_order_gradient(self, y_hat: np.ndarray, y: np.ndarray):
        """
        return first order gradient for the specified var vector
        :param y_hat:
        :param y:
        :param var:
        :return:
        """
        y_hat[y_hat == 1.0] = 0.99999
        y_hat[y_hat == 0.0] = 0.00001
        y_hat, y = reshape_outputs(y_hat, y)
        grad = np.transpose(y_hat-y)
        return grad

    def second_order_gradient(self, y_hat: np.ndarray, y: np.ndarray):
        pass
