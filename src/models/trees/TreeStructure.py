

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
