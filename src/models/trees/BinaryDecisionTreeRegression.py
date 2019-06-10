import numpy as np
import pandas as pd
from typing import Union

from src.models.Model import Model
from src.cost_functions.Cost import Cost
from src.models.trees.TreeStructure import BinaryNode, Leaf


class BinaryDecisionTreeRegression(Model):
    """
    binary decision tree without gradient descent
    """
    def __init__(self, cost_function: Cost):
        assert isinstance(cost_function, Cost), "cost_function must be of type Cost!"
        self.cost_function = cost_function
        self.tree = None
        self.train_label = None
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
        self.train_label = train_label
        converged = False
        feature, split_value, _ = self._pick_feature(train_data, train_label,
                                                     indices=np.array([True for _ in range(len(train_label))]))
        left_indices = np.array(train_data[:, feature] < split_value)
        right_indices = np.invert(left_indices)
        self.tree = tree = BinaryNode(left_indices, right_indices, split_value, feature)

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
        """
        pick the feature and split with has the highest information gain
        :param train_data:
        :param train_label:
        :param indices: boolean array which shows which observations are valide for this leaf
        :return:
        """
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
                cost = self._calculate_cost(train_data, train_label)
                # TODO define stopping criteria for split search
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

    def _calculate_cost(self, train_data: np.ndarray, train_label: np.ndarray):
        pass

    def predict(self, data: np.ndarray):
        """
        Return predictions based on the suppliead data
        :param data:
        :return:
        """
        predictions = np.array([np.NaN for _ in data.shape[0]])
        indices = np.array([True for _ in data.shape[0]])
        self._inner_predict(indices, data, predictions, self.tree)
        return predictions

    def _inner_predict(self, indices: np.ndarray, data: np.ndarray, predictions: np.ndarray, node=None):
        """
        inner predict which implements the predict structure
        :param node:
        :param indices:
        :param data:
        :param predictions:
        :return:
        """
        if node is None:
            predictions = np.mean(self.train_label)
        else:
            left_indices = data[:, node.variable] < node.split_value
            right_indices = np.invert(left_indices)
            left_indices = np.logical_and(left_indices, indices)
            right_indices = np.logical_and(right_indices, indices)
            if node.left_leaf.terminal:
                pred_left = node.left_leaf.prediction
                predictions[left_indices] = pred_left
            else:
                self._inner_predict(left_indices, data, predictions, node.left_leaf.node)

            if node.right_leaf.terminal:
                pred_right = node.right_leaf.prediction
                predictions[right_indices] = pred_right
            else:
                self._inner_predict(right_indices, data, predictions, node.left_leaf.node)

    def print_tree(self):
        pass
