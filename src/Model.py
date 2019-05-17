import abc
import numpy as np
import pandas as pd
from typing import Union


class Model(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self, train_data: Union[pd.DataFrame, np.ndarray], train_label: Union[pd.DataFrame, np.ndarray],
              val_data: Union[pd.DataFrame, np.ndarray], val_label: Union[pd.DataFrame, np.ndarray]):
        pass

    @abc.abstractmethod
    def predict(self, test_data: Union[pd.DataFrame, np.ndarray]):
        pass