import numpy as np


class BinaryNode:
    """
    Helper class which implements a node structure
    """
    def __init__(self, left_indices: np.ndarray, right_indices: np.ndarray, split_value: float, variable: int,
                 prediction_left: float, prediction_right: float, gain: float = None ):
        """
        initialize an binary node
        :param left_indices: indices that belong to the left split
        :param right_indices: indices that belong to the right split
        :param split_value: value for which the chosen variable will be splitted
        """
        assert isinstance(left_indices, np.ndarray)
        assert isinstance(right_indices, np.ndarray)
        assert isinstance(split_value, float)
        assert isinstance(variable, int)
        assert isinstance(prediction_left, float)
        assert isinstance(prediction_right, float)
        assert isinstance(gain, float)
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
