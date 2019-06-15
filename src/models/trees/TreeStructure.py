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
        if self.left_leaf is not None:
            if self.left_leaf.terminal:
                left_leafs = {"l": self.left_leaf}
            else:
                left_leafs = self.left_leaf.node.leafs()
                left_leafs = {"l" + key: value for key, value in left_leafs.items()}
        if self.right_leaf is not None:
            if self.right_leaf.terminal:
                right_leafs = {"r": self.left_leaf}
            else:
                right_leafs = self.right_leaf.node.leafs()
                right_leafs = {"r" + key: value for key, value in right_leafs.items()}
        return left_leafs.update(right_leafs)

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
        for dir in leaf_order:
            if dir == "left":
                leaf = node.left_leaf
            else:
                leaf = node.right_leaf
            if not leaf.terminal:
                node = leaf.node

        return leaf



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
