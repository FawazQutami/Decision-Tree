# File name: DecisionTreeClassifier.py

from collections import Counter
import time

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def entropy(y):
    """
    Entropy - Shannon
    :param y: {array-like}
    :return: {float}
    """
    """
    In information theory, the entropy of a random variable is the average level of "information", 
    "surprise", or "uncertainty" inherent in the variable's possible outcomes. 
    As an example, consider a biased coin with probability p of landing on heads and probability 1-p of 
    landing on tails. The maximum surprise is for p = 1/2, when there is no reason to expect one outcome 
    over another, and in this case a coin flip has an entropy of one bit. 
    The minimum surprise is when p = 0 or p = 1, when the event is known and the entropy is zero bits. 
    Other values of p give different entropies between zero and one bits.
    
    Given a discrete random variable X, with possible outcomes x1,..., xn, which occur with probability 
    P(x1),..., P(xn), the entropy of X is formally defined as:
            H(X) = - ∑i=1:n P(xi)log(P(xi))
    """
    # Count number of occurrences of each value in array of non-negative ints - class labels occurrences
    label_occurrences = np.bincount(y)
    # P(X) = number of all class labels occurrences / total number of samples
    p_x = label_occurrences / len(y)
    # Calculate the Entropy
    ent = -np.sum([p * np.log2(p) for p in p_x if p > 0])
    # Return the entropy - float
    return ent


def most_common_class(y):
    """
    most_common_class method
    :param y: {array-like}
    :return: {int}
    """
    common_class = Counter(y)
    # Get a list of tuple of most common labels
    most_common_class_list = common_class.most_common(1)
    # Return the first tuple and then the first dimension
    most_common = most_common_class_list[0][0]
    return most_common


class Node:
    """ Node - store all the node information"""

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        """
        Class constructor
        :param feature_index: {int} best split feature index
        :param threshold: {float} best split threshold
        :param left: {node} left child tree
        :param right: {node} right child tree
        :param value: {int} common class label for the leaf node
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        # If there is a value then it is a leaf node
        return self.value is not None


class DecisionTreeClassifier:
    """ Decision Tree Classifier """
    """
    --> From Wikipedia
    Decision tree learning is one of the predictive modelling approaches used in statistics,
    data mining and machine learning.
    It uses a decision tree (as a predictive model) to go from observations about an item
    (represented in the branches) to conclusions about the item's target value (represented in
    the leaves).
    Tree models where the target variable can take a discrete set of values are called classification
    trees; in these tree structures, leaves represent class labels and branches represent conjunctions
    of features that lead to those class labels. Decision trees where the target variable can take
    continuous values (typically real numbers) are called regression trees.
    Decision trees are among the most popular machine learning algorithms given their intelligibility
    and simplicity.
    """

    def __init__(self, min_training_samples=2, max_depth=0):
        """
        Class constructor
        :param min_training_samples: {int} minimum number of training samples to use on each leaf
        :param max_depth: {int} Maximum depth refers to the the length of the longest path from
                                a root to a leaf.
        """
        # Set a minimum number of training samples to use on each leaf
        self.min_training_samples = min_training_samples
        # set maximum depth of your model. Maximum depth refers to the the length of
        # the longest path from a root to a leaf.
        self.max_depth = max_depth

        # Root Node - We need to know where we should start the traversing
        self.root = None

    def fit(self, X, y):
        """
        fit Method
        :param X: {array-like}
        :param y: {array-like}
        :return: None
        """
        # Initialize number of features
        self.n_features = X.shape[1]
        # Growing a tree starting from root
        self.root = self.grow_tree(X, y)

    def predict(self, X):
        """
        predict method
        :param X: {array-like}
        :return: {array-like}
        """
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, X, node):
        """
        traverse_tree method
        :param X: {array-like}
        :param node: {root tree}
        :return:
        """
        if node.is_leaf_node():
            return node.value

        if X[node.feature_index] <= node.threshold:
            return self.traverse_tree(X, node.left)
        return self.traverse_tree(X, node.right)

    def grow_tree(self, X, y, depth=0):
        """
        grow_tree method
        :param X: {array-like}
        :param y: {array-like}
        :param depth: {int}
        :return: {Node}
        """
        """
            A tree is built by splitting the source set, constituting the root node of the tree, 
            into subsets—which constitute the successor children. 
            The splitting is based on a set of splitting rules based on classification features.
            This process is repeated on each derived subset in a recursive manner called recursive partitioning. 
            The recursion is completed when the subset at a node has all the same values of the target variable, 
            or when splitting no longer adds value to the predictions. 
            This process of top-down induction of decision trees (TDIDT) is an example of a greedy algorithm,
            and it is by far the most common strategy for learning decision trees from data.
        """
        # Get number of samples
        n_samples = X.shape[0]
        # Get the labels/classes
        n_classes = len(np.unique(y))  # n_classes = len(set(y))

        # When to stop growing a tree? (stopping criteria) - to avoid over-fitting
        if (depth >= self.max_depth  # Check if reached max depth
                or n_classes == 1  # Check if no more class labels
                or n_samples < self.min_training_samples  # Check if min samples exist in Node
                ):
            # If one of the above checks satisfied then:
            # Get the common class in the Node
            leaf_node_value = most_common_class(y)
            # Return the class label as the value of the leaf Node
            return Node(value=leaf_node_value)

        # Otherwise: -----------
        # Greedy Search  - select the best feature index and best threshold according to information gain
        best_feature_index, best_threshold = self.greedy_search(X, y)

        # Split the node samples to left child and right child
        left_indices, right_indices = self.split_node(X[:, best_feature_index], best_threshold)
        x_left, y_left = X[left_indices, :], y[left_indices]
        x_right, y_right = X[right_indices, :], y[right_indices]

        # Grow the children
        left_node = self.grow_tree(x_left, y_left, depth + 1)
        right_node = self.grow_tree(x_right, y_right, depth + 1)

        # Return node information
        return Node(best_feature_index, best_threshold, left_node, right_node)

    def greedy_search(self, X, y):
        """
        greedy_search method
        :param X: {array-like}
        :param y: {array-like}
        :return: {int}: best_feature_index, {float}: best_threshold
        """
        best_gain = -1
        best_feature_index, best_threshold = None, None

        # Loop over all features
        for feature_index in range(self.n_features):
            # Select the vector column (feature) of X by index - one feature array
            x_vector = X[:, feature_index]
            # Get all the possible threshold of the selected feature
            thresholds = np.unique(x_vector)
            # Loop over all thresholds
            for threshold in thresholds:
                # Calculate the information gain
                gain = self.information_gain(y, x_vector, threshold)
                # Check if the gain is the best gain
                if gain > best_gain:
                    # Best gain is the gain
                    best_gain = gain
                    # Save the index and the threshold
                    best_feature_index = feature_index
                    best_threshold = threshold

        # Return the best feature and the best threshold
        return best_feature_index, best_threshold

    def information_gain(self, y, x_vector, threshold):
        """
        information_gain method
        :param y: {array-like}
        :param x_vector: {array-like} one feature array
        :param threshold: {float}
        :return: {float}: info_gain
        """
        # Calculate the parent entropy
        parent = entropy(y)

        # Get the indices by splitting the node to left and right
        left_indices, right_indices = self.split_node(x_vector, threshold)
        # Check on the indices
        if (len(left_indices) == 0
                or len(right_indices) == 0):
            return 0
        # Get the total number of samples
        n_samples = len(y)
        # Get the number of samples of the left child and right child
        n_samples_left, n_samples_right = len(left_indices), len(right_indices)

        # Calculate the left children entropy
        l_child_entropy = entropy(y[left_indices])
        # Calculate the right children entropy
        r_child_entropy = entropy(y[right_indices])

        # Calculate the weighted average of the entropy of the children
        children = (n_samples_left / n_samples) * l_child_entropy \
                   + (n_samples_right / n_samples) * r_child_entropy

        # Calculate information gain - difference in loss
        info_gain = parent - children

        # Return information gain
        return info_gain

    def split_node(self, x_vector, threshold):
        """
        split_node method
        :param x_vector: {array-like}
        :param threshold: {float}
        :return: {array-like}: left_indices, {array-like}: right_indices
        """
        # Find the non-zero grouped elements of the left node indices if the vector column
        # is less or equal the threshold
        left_indices = np.argwhere(x_vector <= threshold).flatten()     # flatten to get 1d array
        # Find the non-zero grouped elements of the right node indices if the vector column
        # is bigger than the threshold
        right_indices = np.argwhere(x_vector > threshold).flatten()

        # Return the left and right indices
        return left_indices, right_indices


if __name__ == '__main__':
    # Load data
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1234)

    start = time.time()

    d_tree = DecisionTreeClassifier(max_depth=10)
    d_tree.fit(X_train, y_train)

    y_prediction = d_tree.predict(X_test)
    print('Accuracy is: %.2f%%' % (np.sum(y_test == y_prediction) / len(y_test) * 100))

    end = time.time()  # ----------------------------------------------
    print('\n ----------\n Execution Time: {%f}' % ((end - start) / 1000) + ' seconds.')
