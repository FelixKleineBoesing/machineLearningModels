import numpy as np
from typing import Union


class BinaryNode:
    """
    Helper class which implements a node structure
    """
    def __init__(self, left_indices: np.ndarray, right_indices: np.ndarray,
                 split_value: Union[float, np.float64, np.int64],
                 variable: int, prediction_left: Union[float, np.float64], prediction_right: Union[float, np.float64],
                 gain: Union[float, np.float64] = None):
        """
        initialize an binary node
        :param left_indices: indices that belong to the left split
        :param right_indices: indices that belong to the right split
        :param split_value: value for which the chosen variable will be splitted
        """
        assert isinstance(left_indices, np.ndarray)
        assert isinstance(right_indices, np.ndarray)
        assert isinstance(variable, int)
        assert type(split_value) in [float, np.float, np.float64, np.int64]
        assert type(prediction_left) in [float, np.float, np.float64, np.int64]
        assert type(prediction_right) in [float, np.float, np.float64, np.int64]
        assert type(gain) in [float, np.float, np.float64]
        self.left_leaf = Leaf(left_indices, prediction_left)
        self.right_leaf = Leaf(right_indices, prediction_right)
        self.split_value = split_value
        self.variable = variable
        self.gain = gain

    def leafs(self):
        """
        return leafs of tree here
        :return: I don´t know, have to think about this
        """
        if self.left_leaf is not None:
            if self.left_leaf.terminal:
                left_leafs = {"l": self.left_leaf}
            else:
                left_leafs = self.left_leaf.node.leafs()
                left_leafs = {"l" + key: value for key, value in left_leafs.items()}
        if self.right_leaf is not None:
            if self.right_leaf.terminal:
                right_leafs = {"r": self.right_leaf}
            else:
                right_leafs = self.right_leaf.node.leafs()
                right_leafs = {"r" + key: value for key, value in right_leafs.items()}
        left_leafs.update(right_leafs)
        return left_leafs

    def depth(self):
        if not self.left_leaf.terminal:
            left_depth = self.left_leaf.node.depth()
        else:
            left_depth = 1
        if not self.right_leaf.terminal:
            right_depth = self.right_leaf.node.depth()
        else:
            right_depth = 1
        return max(left_depth, right_depth)

    def get_leaf(self, leaf_order: tuple):
        """
        get leaf based on the order of right/left
        :param leaf_order:
        :return:
        """
        node = self
        for index, dir in enumerate(leaf_order):
            if dir == "l":
                leaf = node.left_leaf
            else:
                leaf = node.right_leaf
            node = leaf.node

        return leaf

    def get_tree_structure(self) -> dict:
        leafs = {"name": "{0} < {1}".format(self.variable, self.split_value), "children": []}
        if self.left_leaf.terminal:
            leafs["children"].append({"name":"Prediction: {0},    Anzahl Obs: {1}".format(self.left_leaf.prediction,
                                                                             sum(self.left_leaf.indices))})
        else:
            leafs["children"].append(self.left_leaf.node.get_tree_structure())
        if self.right_leaf.terminal:
            leafs["children"].append({"name":"Prediction: {0},    Anzahl Obs: {1}".format(self.right_leaf.prediction,
                                                                             sum(self.right_leaf.indices))})
        else:
            leafs["children"].append(self.right_leaf.node.get_tree_structure())
        return leafs


class Leaf:

    def __init__(self, indices: np.ndarray, prediction: float):
        """

        :param indices: indices of train data
        :param value: value must be the prediction for this leaf
        """
        assert isinstance(indices, np.ndarray)
        assert isinstance(prediction, float)
        self._node = None
        self._terminal = True
        self.indices = indices
        self.prediction = prediction

    @property
    def node(self):
        return self._node

    @node.setter
    def node(self, node: BinaryNode):
        assert isinstance(node, BinaryNode)
        self._node = node
        self._terminal = False

    @property
    def terminal(self):
        return self._terminal


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
            id_ = None
            for key, value in self._leafs.items():
                if value["gain"] > gain:
                    id_ = key
            return id_, self._leafs[id_]
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

