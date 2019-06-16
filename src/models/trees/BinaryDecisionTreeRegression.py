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
    def __init__(self, cost_function: Cost, params: dict = None):
        assert isinstance(cost_function, Cost), "cost_function must be of type Cost!"
        self.cost_function = cost_function
        self.tree = None
        self.train_label = None
        if params is None:
            params = {}

        if "max_depth" not in params:
            params["max_depth"] = 8
        self.params = params

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
        stopped = False
        max_depth = self.params["max_depth"]
        feature, split_value, gain = self._pick_feature(train_data, train_label,
                                                        indices=np.array([True for _ in range(len(train_label))]))
        left_indices = np.array(train_data[:, feature] < split_value)
        right_indices = np.invert(left_indices)
        self.tree = BinaryNode(left_indices=left_indices, right_indices=right_indices, split_value=split_value,
                               variable=feature, prediction_left=np.mean(train_label[left_indices]),
                               prediction_right=np.mean(train_label[right_indices]), gain=gain)

        leafs = LeafStorer()
        while not stopped:
            all_leafs = self.tree.leafs()
            max_depth_reached = True
            for key, leaf in all_leafs.items():
                if len(key) >= max_depth:
                    continue
                if key in leafs:
                    max_depth_reached = False
                    continue
                max_depth_reached = False
                feature, split_value, gain = self._pick_feature(train_data, train_label, leaf.indices)
                leafs[key] = {"feature": feature, "split_value": split_value,
                              "gain": gain}

            id_, leaf_split = leafs.pop_max_gain_leaf()

            feature = leaf_split["feature"]
            split_value = leaf_split["split_value"]
            gain = leaf_split["gain"]
            leaf = self.tree.get_leaf(id_)
            bool_ = train_data[:, feature] < split_value
            left_indices = np.logical_and(leaf.indices, bool_)
            right_indices = np.logical_and(leaf.indices, np.invert(bool_))
            leaf.node = BinaryNode(left_indices=left_indices, right_indices=right_indices, split_value=split_value,
                                   variable=feature, prediction_left=float(np.mean(train_label[left_indices])),
                                   prediction_right=float(np.mean(train_label[right_indices])), gain=gain)
            if max_depth_reached:
                stopped = True

    def _pick_feature(self, train_data: np.ndarray, train_label: np.ndarray, indices: np.ndarray):
        """
        pick the feature and split with has the highest information gain
        :param train_data:
        :param train_label:
        :param indices: boolean array which shows which observations are valide for this leaf
        :return:
        """
        prediction = self.predict(train_data)
        cost = self.cost_function.compute(prediction[indices], train_label[indices])
        feature = None
        min_cost = np.inf
        chosen_split_value = None
        for i in range(train_data.shape[1]):
            unique_values = np.sort(np.unique(train_data[:, i]))
            for split_value in unique_values:
                left_indices = np.array(train_data[:, i] < split_value)
                right_indices = np.invert(left_indices)
                predictions = np.zeros(len(indices))
                predictions[left_indices] = np.mean(train_label[left_indices])
                predictions[right_indices] = np.mean(train_label[right_indices])
                cost_split = self.cost_function.compute(predictions, train_label)
                if cost_split < min_cost:
                    feature, chosen_split_value = i, split_value
                    min_cost = cost_split
        gain = cost - min_cost
        return feature, chosen_split_value, gain

    def predict(self, data: np.ndarray):
        """
        Return predictions based on the suppliead data
        :param data:
        :return:
        """
        predictions = np.array([np.NaN for _ in range(data.shape[0])])
        indices = np.array([True for _ in range(data.shape[0])])
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
            predictions[:] = np.mean(self.train_label)
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


class LeafStorer:
    """
    Used in building the tree. We store already calculated splits here for later use.
    We store only best split per leaf.
    """

    def __init__(self):
        self._leafs = {}
        self.max_gain = np.inf * -1
        self._gains = []

    def __setitem__(self, key, value):
        self.max_gain = max(self.max_gain, value["gain"])
        self._gains.append(value["gain"])
        self._leafs[key] = value

    def __getitem__(self, item):
        return self._leafs[item]

    def __len__(self):
        return len(self._leafs)

    def __contains__(self, key):
        return key in self._leafs

    def get_max_gain_leaf(self):
        if len(self._leafs) > 0:
            gain = -1 * np.Inf
            id = None
            for key, value in self._leafs.items():
                if value["gain"] > gain:
                    id = key
            return id, self._leafs[id]
        else:
            raise IndexError("No data present in LeafStorer, can´t retrieve max gain split")

    def delete_item(self, key):
        del self._leafs[key]
        del self._gains[self._gains.index(max(self._gains))]
        if len(self._gains) > 0:
            self.max_gain = max(self._gains)

    def pop_max_gain_leaf(self):
        id, leaf = self.get_max_gain_leaf()
        self.delete_item(id)
        return id, leaf