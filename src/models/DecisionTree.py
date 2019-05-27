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
        converged = False

        while not converged:
            feature = self._pick_feature()
            split_value = self._find_split()



    def _pick_feature(self, data: np.ndarray):
        pass

    def _find_split(self, data: np.ndarray):
        pass

    def _detect_convergence(self):
        pass

    def _detect_overfitting(self):
        pass

    def _prune_tree(self):
        pass

    def predict(self, test_data: Union[pd.DataFrame, np.ndarray]):
        pass


class Node:

    def __init__(self):
        self.left = None
        self.right = None
        self.terminal = False
        self.indices = []
