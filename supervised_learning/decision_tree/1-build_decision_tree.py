#!/usr/bin/env python3

class Node:
    """A class representing a node in a decision tree.
    
    Contains information about the splitting feature, threshold value,
    and references to child nodes.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=0, is_root=False):
        """Initialize a new Node.

        Args:
            feature: Feature index used for splitting
            threshold: Value to split on
            left_child: Left child node
            right_child: Right child node
            depth: Depth of node in tree
            is_root: Boolean indicating if node is root
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes in the subtree rooted at this node.

        Args:
            only_leaves: If True, count only leaf nodes

        Returns:
            int: Number of nodes (or leaves) in subtree
        """
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                   self.right_child.count_nodes_below(only_leaves=True))
        return (1 + self.left_child.count_nodes_below(only_leaves=False) +
               self.right_child.count_nodes_below(only_leaves=False))


class Leaf:
    """A class representing a leaf node in a decision tree.
    
    Contains the predicted value for instances reaching this leaf.
    """

    def __init__(self, value, depth=0):
        """Initialize a new Leaf node.

        Args:
            value: Predicted value at this leaf
            depth: Depth of leaf in tree
        """
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below this leaf (always returns 1).

        Args:
            only_leaves: Ignored for leaf nodes

        Returns:
            int: Always returns 1 as leaves have no children
        """
        return 1


class Decision_Tree:
    """A class representing a complete decision tree.
    
    Manages the root node and provides tree-level operations.
    """

    def __init__(self, root):
        """Initialize a new Decision Tree.

        Args:
            root: Root node of the tree
        """
        self.root = root

    def count_nodes(self, only_leaves=False):
        """Count total nodes in the tree.

        Args:
            only_leaves: If True, count only leaf nodes

        Returns:
            int: Total number of nodes (or leaves) in tree
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
