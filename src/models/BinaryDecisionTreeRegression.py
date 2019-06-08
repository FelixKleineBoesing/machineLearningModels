import numpy as np
import pandas as pd
from typing import Union

from src.models.Model import Model


class BinaryDecisionTreeRegression(Model):
    """
    binary decision tree without gradient descent
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
        feature, split_value, _ = self._pick_feature(train_data, train_label)
        left_indices = np.array(train_data[:, feature] < split_value)
        right_indices = np.invert(left_indices)
        tree = BinaryNode(left_indices, right_indices, split_value, feature)
        while not converged:
            features = []
            split_values = []
            gains = []
            leafs = []
            for leaf in tree.leafs():
                feature, split_value, gain = self._pick_feature(train_data, train_label, )
                features.append(feature)
                split_values.append(split_value)
                gains.append(gain)
                leafs.append(leaf)
            max_gain_index = gains.index(max(gains))
            feature = features[max_gain_index]
            gain = gains[max_gain_index]
            split_value = split_values[max_gain_index]
            leaf = leafs[max_gain_index]
            # TODO add found split to tree

    def _pick_feature(self, train_data: np.ndarray, train_label: np.ndarray, indices: np.ndarray):
        feature = None
        min_cost = np.inf
        chosen_split_value = None
        for i in train_data.shape[1]:
            optimal_split = False
            unique_values = np.sort(np.unique(train_data[:, i]))
            rel_index = 0.5
            while not optimal_split:
                split_value = unique_values[int(rel_index*len(unique_values))]
                left_indices = np.array(train_data[:, i] < split_value)
                right_indices = np.invert(left_indices)
                # TODO evaluate split with cost measure
                cost = self._calculate_cost(train_data, train_label, feature, split_value)
                # TODo define stopping criteria for split search
                if cost < min_cost:
                    feature, chosen_split_value = i, split_value

        return feature, chosen_split_value, gain

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
    def __init__(self, left_indices: np.ndarray, right_indices: np.ndarray, split_value: float, variable: int):
        """
        initialize an binary node
        :param left_indices: indices that belong to the left split
        :param right_indices: indices that belong to the right split
        :param split_value: value for which the chosen variable will be splitted
        """
        self.left_leaf = Leaf(left_indices)
        self.right_leaf = Leaf(right_indices)
        self.split_value = split_value
        self.variable = variable

    def leafs(self):
        """
        return leafs of tree here
        :return: I don´t know, have to think about this
        """
        leafs = []
        if self.left_leaf is not None:
            if self.left_leaf.terminal:
                leafs.append(self.left_leaf)
            else:
                leafs += self.left_leaf.node.leafs()
        if self.right_leaf is not None:
            if self.right_leaf.terminal:
                leafs.append(self.right_leaf)
            else:
                leafs += self.right_leaf.node.leafs()
        return leafs

    def get_depth(self):
        pass


class Leaf:

    def __init__(self, indices: np.ndarray):
        self._node = None
        self._terminal = True
        self._indices = indices
        self.prediction = None

    @property
    def node(self):
        return self._node

    @node.setter
    def node(self, node: BinaryNode):
        self._node = node
        self._terminal = False

    @property
    def terminal(self):
        return self._terminal
