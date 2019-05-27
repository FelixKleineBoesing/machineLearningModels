import numpy as np
import pandas as pd
from typing import Union

from src.models.Model import Model


class DecisionTree(Model):
    """
    implement CHAID Trees
    """
    def __init__(self):
        super().__init__()

    def train(self, train_data: Union[pd.DataFrame, np.ndarray], train_label: Union[pd.DataFrame, np.ndarray],
              val_data: Union[pd.DataFrame, np.ndarray], val_label: Union[pd.DataFrame, np.ndarray]):
        pass

    def predict(self, test_data: Union[pd.DataFrame, np.ndarray]):
        pass
