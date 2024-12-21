#!/usr/bin/env python3
"documentation documentation"


class Node:
    "documentation documentation"
    def __init__(self, feature, threshold, left_child, right_child, depth=0, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root

    def count_nodes_below(self, only_leaves=False):
        # If only counting leaves, count leaves in left and right subtrees
        if only_leaves:
            return self.left_child.count_nodes_below(only_leaves=True) + self.right_child.count_nodes_below(only_leaves=True)
        # Otherwise, count this node and nodes in its subtrees
        return 1 + self.left_child.count_nodes_below(only_leaves=False) + self.right_child.count_nodes_below(only_leaves=False)


class Leaf:
    "documentation documentation"
    def __init__(self, value, depth=0):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def count_nodes_below(self, only_leaves=False):
        # A leaf is always counted as one node
        return 1


class Decision_Tree:
    "documentation documentation"
    def __init__(self, root):
        "documentation documentation"
        self.root = root

    def count_nodes(self, only_leaves=False):
        "documentation documentation"
        return self.root.count_nodes_below(only_leaves=only_leaves)