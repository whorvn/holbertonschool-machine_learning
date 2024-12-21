#!/usr/bin/env python3
"""
Defines classes for constructing and managing a decision tree.

Includes Node, Leaf, and Decision_Tree classes with pretty-printing
support for tree structure visualization.
"""


def left_child_add_prefix(text):
    """
    Adds a left child prefix to the provided text.
    """
    lines = text.split("\n")
    new_text = "    +---> " + lines[0] + "\n"
    for line in lines[1:]:
        new_text += "    |      " + line + "\n"
    return new_text


def right_child_add_prefix(text):
    """
    Adds a right child prefix to the provided text.
    """
    lines = text.split("\n")
    new_text = "    +---> " + lines[0] + "\n"
    for line in lines[1:]:
        new_text += "           " + line + "\n"
    return new_text


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

    def __str__(self):
        """
        Returns a string representation of the node and its children.
        """
        text = f"node [feature={self.feature}, threshold={self.threshold}]"
        left_str = left_child_add_prefix(str(self.left_child))
        right_str = right_child_add_prefix(str(self.right_child))
        return text + "\n" + left_str + right_str


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

    def __str__(self):
        """
        Returns a string representation of the leaf.
        """
        return f"leaf [value={self.value}]"


class Decision_Tree:
    """
    Represents a decision tree with a root node.
    """

    def __init__(self, root):
        """
        Initializes the decision tree.
        """
        self.root = root

    def __str__(self):
        """
        Returns a string representation of the entire decision tree.
        """
        return f"root {str(self.root)}"
