import numpy as np


def reshape_outputs(y_hat, y):
    y_hat = y_hat.copy()
    y = y.copy()
    if len(y_hat.shape) == 1:
        y_hat = y_hat.reshape(y_hat.shape[0], 1)
    if len(y.shape) == 1:
        y = y.reshape(y.shape[0], 1)
    return y_hat, y
