import numpy as np


def mean_sqared_error(y_hat: np.ndarray, y: np.ndarray):
    return sum((y_hat - y) ^ 2) / (2*len(y))
