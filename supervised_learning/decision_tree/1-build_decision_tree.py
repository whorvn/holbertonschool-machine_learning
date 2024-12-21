#!/usr/bin/env python3

class Node:
    """
    Represents an internal node in a decision tree.

    Attributes:
        feature (int): The feature index used for splitting.
        threshold (float): The threshold value for the split.
        left_child (Node or Leaf): The left child of the node.
        right_child (Node or Leaf): The right child of the node.
        depth (int): The depth of the node in the tree.
        is_root (bool): Indicates whether the node is the root of the tree.
        is_leaf (bool): Always False for internal nodes.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=0, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes below this node, including or excluding internal nodes.

        Args:
            only_leaves (bool): Whether to count only leaf nodes.

        Returns:
            int: The count of nodes below this node.
        """
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                    self.right_child.count_nodes_below(only_leaves=True))
        return (1 + self.left_child.count_nodes_below(only_leaves=False) +
                self.right_child.count_nodes_below(only_leaves=False))


class Leaf:
    """
    Represents a leaf node in a decision tree.

    Attributes:
        value (int or float): The value stored in the leaf.
        depth (int): The depth of the leaf in the tree.
        is_leaf (bool): Always True for leaf nodes.
    """

    def __init__(self, value, depth=0):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes below this leaf. Always returns
        1 as leaves have no children.

        Args:
            only_leaves (bool): Ignored for leaves.

        Returns:
            int: Always 1.
        """
        return 1


class Decision_Tree:
    """
    Represents a decision tree.

    Attributes:
        root (Node or Leaf): The root node of the decision tree.
    """

    def __init__(self, root):
        self.root = root

    def count_nodes(self, only_leaves=False):
        """
        Count nodes in the decision tree, including
        or excluding internal nodes.

        Args:
            only_leaves (bool): Whether to count only leaf nodes.

        Returns:
            int: The count of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
