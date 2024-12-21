#!/usr/bin/env python3

class Node:
    """documentation for doc"""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=0, is_root=False):
        """documentation for doc"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def count_nodes_below(self, only_leaves=False):
        """documentation for doc"""
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                    self.right_child.count_nodes_below(only_leaves=True))
        return (1 + self.left_child.count_nodes_below(only_leaves=False) +
                self.right_child.count_nodes_below(only_leaves=False))


class Leaf:
    """documentation for doc"""

    def __init__(self, value, depth=0):
        """documentation for doc"""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def count_nodes_below(self, only_leaves=False):
        """documentation for doc"""
        return 1


class Decision_Tree:
    """documentation for doc"""

    def __init__(self, root):
        """documentation for doc"""
        self.root = root

    def count_nodes(self, only_leaves=False):
        """documentation for doc"""
        return self.root.count_nodes_below(only_leaves=only_leaves)
