import numpy as np
import pandas as pd
from typing import Union

from src.models.Model import Model


class BinaryDecisionTreeRegression(Model):
    """
    implement CHAID Trees
    """
    def __init__(self):
        super().__init__()

    def train(self, train_data: Union[pd.DataFrame, np.ndarray], train_label: Union[pd.DataFrame, np.ndarray],
              val_data: Union[pd.DataFrame, np.ndarray], val_label: Union[pd.DataFrame, np.ndarray]):
        """
        builds a decision tree
        :param train_data:
        :param train_label:
        :param val_data:
        :param val_label:
        :return:
        """
        converged = False
        feature, split_value = self._pick_feature(train_data, train_label)
        tree = BinaryNode()
        while not converged:
            for leafs in tree.leafs():
                feature, split_value = self._pick_feature()


            # TODO add tree structure


    def _pick_feature(self, train_data: np.ndarray, train_label: np.ndarray):
        n = train_data.shape[0]
        feature = None
        min_cost = np.inf
        chosen_split_value = None
        for i in train_data.shape[1]:
            optimal_split = False
            unique_values = np.sort(np.unique(train_data[:, i]))
            rel_index = 0.5
            while not optimal_split:
                split_value = unique_values[int(rel_index*len(unique_values))]
                left = train_data[train_data[:,i] < split_value]
                right = train_data[train_data[:, i] >= split_value]
                # TODO evaluate split with cost measure
                cost = self._calculate_cost(train_data, train_label, feature, split_value)
                # TODo define stopping criteria for split search
                if cost < min_cost:
                    feature, chosen_split_value = i, split_value

        return feature, chosen_split_value

    def _find_split(self, data: np.ndarray):
        pass

    def _detect_convergence(self):
        pass

    def _detect_overfitting(self):
        pass

    def _prune_tree(self):
        pass

    def _calculate_cost(self, train_data: np.ndarray, train_label: np.ndarray, feature: int, split_value: float):
        pass

    def predict(self, data: np.ndarray):
        pass

    def print_tree(self):
        pass


class BinaryNode:
    """
    Helper class which implements a node structure
    """
    def __init__(self, left_indices: list, right_indices: list, split_value: float, variable: int):
        """
        initialize an binary node
        :param left_indices: indices that belong to the left split
        :param right_indices: indices that belong to the right split
        :param split_value: value for which the chosen variable will be splitted
        """
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.split_value = split_value
        self.variable = variable
        self.left_terminal = False
        self.right_terminal = False

