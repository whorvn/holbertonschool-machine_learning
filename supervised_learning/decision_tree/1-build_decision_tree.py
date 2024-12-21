#!/usr/bin/env python3

class Node:
    """
    Represents an internal node in a decision tree.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=0, is_root=False):
        """
        Initializes a decision tree node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def count_nodes_below(self, only_leaves=False):
        """
        Counts nodes below, optionally only leaves.
        """
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                    self.right_child.count_nodes_below(only_leaves=True))
        return (1 + self.left_child.count_nodes_below(only_leaves=False) +
                self.right_child.count_nodes_below(only_leaves=False))


class Leaf:
    """
    Represents a leaf node in a decision tree.
    """

    def __init__(self, value, depth=0):
        """
        Initializes a leaf node.
        """
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def count_nodes_below(self, only_leaves=False):
        """
        Counts nodes below, always 1 for leaves.
        """
        return 1


class Decision_Tree:
    """
    Represents a decision tree with a root node.
    """

    def __init__(self, root):
        """
        Initializes the decision tree.
        """
        self.root = root

    def count_nodes(self, only_leaves=False):
        """
        Counts nodes, optionally only leaves.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
