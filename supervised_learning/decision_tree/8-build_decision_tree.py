#!/usr/bin/env python3
""" Task 8: 8. Using Gini impurity function as a splitting criterion """
import numpy as np


def left_child_add_prefix(text):
    """
    documentation documentation
    """
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("    |  "+x) + "\n"
    return new_text


def right_child_add_prefix(text):
    """
    documentation documentation
    """
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("       " + x) + "\n"
    return new_text


class Node:
    """
    documentation documentation
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """
    documentation documentation
    """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Calculates the maximum depth of the subtree rooted at this node.
        """
        if self.is_leaf:
            return self.depth
        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        else:
            left_depth = self.depth
        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        else:
            right_depth = self.depth
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes in the subtree rooted at this node.
        """
        if self.is_leaf:
            return 1
        if self.left_child:
            left_count = self.left_child.count_nodes_below(only_leaves)
        else:
            left_count = 0
        if self.right_child:
            right_count = self.right_child.count_nodes_below(only_leaves)
        else:
            right_count = 0
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def __str__(self):
        """
        Returns a string representation of the node and its children.
        """
        if self.is_root:
            Type = "root "
        elif self.is_leaf:
            return f"-> leaf [value={self.value}]"
        else:
            Type = "-> node "
        if self.left_child:
            left_str = left_child_add_prefix(str(self.left_child))
        else:
            left_str = ""
        if self.right_child:
            right_str = right_child_add_prefix(str(self.right_child))
        else:
            right_str = ""
        return f"{Type}[feature={self.feature}, threshold=\
{self.threshold}]\n{left_str}{right_str}".rstrip()

    def get_leaves_below(self):
        """
        Returns a list of all leaves below this node.
        """
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Update the bounds for the current node and propagate the
        """
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """
        Compute the indicator function for the current
        """

        def is_large_enough(x):
            """
            Check if each individual has all its features
            """
            return np.all(np.array([x[:, key] > self.lower[key]
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            """
            Check if each individual has all its features
            """
            return np.all(np.array([x[:, key] <= self.upper[key]
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda x: \
            np.all(np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Predict the class for a single individual at the node.
        """
        if self.is_leaf:
            return self.value
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    A class representing a leaf node in a decision tree, inheriting from Node.
    """

    def __init__(self, value, depth=None):
        """
        Initializes a Leaf with the given parameters.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes in the subtree rooted at this leaf.
        """
        return 1

    def __str__(self):
        """
        Returns a string representation of the leaf node.
        """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """
        Returns a list of all leaves below this leaf.
        """
        return [self]

    def update_bounds_below(self):
        """
        Placeholder function for updating the
        """
        pass

    def pred(self, x):
        """
        Predict the class for a single individual at the leaf node.
        """
        return self.value


class Decision_Tree():
    """
    A class representing a decision tree.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initializes a Decision_Tree with the given parameters.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Returns the maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes in the decision tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns a string representation of the decision tree.
        """
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """
        Returns a list of all leaves in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Update the bounds for the entire
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Update the prediction function for the decision tree.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            """
            Predict the class for each individual in the input
            """
            predictions = np.zeros(A.shape[0], dtype=int)
            for i, x in enumerate(A):
                for leaf in leaves:
                    if leaf.indicator(np.array([x])):
                        predictions[i] = leaf.value
                        break
            return predictions
        self.predict = predict

    def pred(self, x):
        """
        Predict the class for a single individual using the decision tree.
        """
        return self.root.pred(x)

    def np_extrema(self, arr):
        """
        Returns the minimum and maximum values of the array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Determines a random split criterion for a given node.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(self.explanatory
                                                       [:, feature]
                                                       [node.sub_population])
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1-x)*feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """
        Fits the decision tree to the provided explanatory and target data.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
            print(f"    - Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def fit_node(self, node):
        """
        Recursively fits the decision tree nodes.
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & \
            (self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & ~left_population
        if len(left_population) != len(self.target):
            left_population = np.pad(left_population,
                                     (0, len(self.target) -
                                      len(self.left_population)),
                                     'constant', constant_values=(0))
        if len(right_population) != len(self.target):
            right_population = np.pad(right_population,
                                      (0, len(self.target) -
                                       len(self.right_population)),
                                      'constant', constant_values=(0))
        is_left_leaf = (node.depth == self.max_depth - 1 or
                        np.sum(left_population) <= self.min_pop or
                        np.unique(self.target[left_population]).size == 1)
        is_right_leaf = (node.depth == self.max_depth - 1 or
                         np.sum(right_population) <= self.min_pop or
                         np.unique(self.target[right_population]).size == 1)
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            node.left_child.depth = node.depth + 1
            self.fit_node(node.left_child)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            node.right_child.depth = node.depth + 1
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Creates a leaf child node.
        """
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates a leaf child node.
        """
        A = self.target[sub_population]
        B, C = np.unique(A, return_counts=True)
        value = B[np.argmax(C)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates a non-leaf child node.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Calculates the accuracy of the decision tree on the test data.
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target))/test_target.size

    def possible_thresholds(self, node, feature):
        """
        Calculate possible thresholds for splitting a decision
        """
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """
        Calculate the Gini impurity for all possible
        """
        thresholds = self.possible_thresholds(node, feature)
        indices = np.arange(self.explanatory.shape[0])[node.sub_population]
        feature_values = self.explanatory[indices, feature]
        target_reduced = self.target[indices]
        classes = np.unique(target_reduced)

        gini_sum = []
        for threshold in thresholds:
            left_indices = feature_values > threshold
            right_indices = ~left_indices

            gini_left, gini_right = 0, 0
            for a in classes:
                p_left = np.mean(target_reduced[left_indices] == a)
                p_right = np.mean(target_reduced[right_indices] == a)
                gini_left += p_left * (1 - p_left)
                gini_right += p_right * (1 - p_right)

            left_size = np.sum(left_indices)
            right_size = np.sum(right_indices)
            total_size = left_size + right_size
            gini_sum1 = ((gini_left * left_size + gini_right * right_size)
                         / total_size)
            gini_sum.append(gini_sum1)

        min_index = np.argmin(gini_sum)
        return np.array([thresholds[min_index], gini_sum[min_index]])

    def Gini_split_criterion(self, node):
        """
        Determine the best feature and its associated
        """
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
