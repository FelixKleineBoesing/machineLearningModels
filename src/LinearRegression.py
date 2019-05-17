import numpy as np
import pandas as pd
from typing import Union
import abc

from src import Model


class LinearRegression(Model):

    def __init__(self, alpha: float = 0.2, iterations: int = 10):
        self.iterations = iterations
        self.alpha = alpha

    def train(self, train_data: Union[pd.DataFrame, np.ndarray], train_label: Union[pd.DataFrame, np.ndarray],
              val_data: Union[pd.DataFrame, np.ndarray], val_label: Union[pd.DataFrame, np.ndarray]):
        pass


    def predict(self, test_data: Union[pd.DataFrame, np.ndarray]):
        pass



class Cost(abc.ABC):


    def __init__(self):
        pass

    def compute(self):
        pass

    def first_order_gradient(self):
        pass

    def second_order_gradient(self):
        pass