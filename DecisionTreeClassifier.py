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


class Node:
    """ Node Leaf - store all the node leaf information"""

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """
        Class constructor
        :param feature: best split feature
        :param threshold: best split threshold
        :param left: left child tree
        :param right: right child tree
        :param value: common class label for the leaf
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        # If there is a value then it is a leaf (node)
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

    def __init__(self, min_training_samples=2, max_depth=100, rf_features=None):
        """
        Class constructor
        :param min_training_samples: {int} minimum number of training samples to use on each leaf
        :param max_depth: {int} Maximum depth refers to the the length of the longest path from
                                a root to a leaf.
        :param rf_features: {None} A random factor
        """
        # Set a minimum number of training samples to use on each leaf
        self.min_training_samples = min_training_samples
        # set maximum depth of your model. Maximum depth refers to the the length of
        # the longest path from a root to a leaf.
        self.max_depth = max_depth

        self.rf_features = rf_features
        # We need to know where we should start the traversing
        self.root = None

    def fit(self, x_trn, y_trn):
        """
        Fit Method
        :param x_trn: {array-like}
        :param y_trn: {array-like}
        :return: None
        """
        # Initialize the random factor with the number of features
        self.rf_features = x_trn.shape[1] if not self.rf_features \
            else min(self.rf_features, x_trn.shape[1])
        # Growing a tree starting from root
        self.root = self.grow(x_trn, y_trn)

    def predict(self, x_test):
        """
        predict method
        :param x_test: {array-like}
        :return: {array-like}
        """
        return np.array([self.traverse_tree(x, self.root) for x in x_test])

    def traverse_tree(self, x, node):
        """
        traverse_tree method
        :param x: {array-like}
        :param node: {root tree}
        :return:
        """
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

    def grow(self, x_trn, y_trn, depth=0):
        """
        grow method
        :param x_trn: {array-like}
        :param y_trn: {array-like}
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
        # Initialize the parameters
        n_samples, n_features = x_trn.shape
        # Get the labels
        n_labels = len(np.unique(y_trn))

        # When to stop growing a tree? (stopping criteria) - to avoid overfitting
        if (depth >= self.max_depth  # Check if reached max depth
                or n_labels == 1  # Check if no more class labels
                or n_samples < self.min_training_samples):  # Check if min samples exist in Node
            # If one of the above checks satisfied then:
            # Get the common class in the Node
            common_class = Counter(y_trn)
            # Get a list of tuple of most common labels
            most_common_class = common_class.most_common(1)
            # Return the first tuple and then the first dimension
            leaf_value = most_common_class[0][0]

            # Return the class label as the value of the leaf Node
            return Node(value=leaf_value)

        # Otherwise:
        # Generate a uniform random sample from np.arange(n_features) of size [self.rf_features]
        # without replacement:
        feature_indices = np.random.choice(n_features, self.rf_features, replace=False)

        # Greedy Search - select the best split according to information gain
        best_feature, best_threshold = self.greedy_search(x_trn, y_trn, feature_indices)

        # Split the node to left child and right child - according to the resulted greedy search split
        left_indices, right_indices = self.split_node(x_trn[:, best_feature], best_threshold)
        x_left, y_left = x_trn[left_indices, :], y_trn[left_indices]
        x_right, y_right = x_trn[right_indices, :], y_trn[right_indices]

        # Grow the children
        left_node = self.grow(x_left, y_left, depth + 1)
        right_node = self.grow(x_right, y_right, depth + 1)

        # Return node information
        return Node(best_feature, best_threshold, left_node, right_node)

    def greedy_search(self, x_trn, y_trn, feature_indices):
        """
        greedy_search method
        :param x_trn: {array-like}
        :param y_trn: {array-like}
        :param feature_indices: {array-like}
        :return: {int}, {int}
        """
        best_gain = -1
        best_feature, best_threshold = None, None

        # Loop over all features
        for feature_index in feature_indices:
            # Select the vector column of X by feature index
            x_vector = x_trn[:, feature_index]
            # Get all the possible threshold of the selected column vector
            thresholds = np.unique(x_vector)

            # Loop over all thresholds
            for threshold in thresholds:
                # Calculate the information gain
                gain = self.information_gain(y_trn, x_vector, threshold)
                # Check if the gain is the best
                if gain > best_gain:
                    # Best gain is the gain
                    best_gain = gain
                    # Save the index and the threshold
                    best_feature = feature_index
                    best_threshold = threshold

        # Return the best feature and the best threshold
        return best_feature, best_threshold

    def information_gain(self, y_trn, x_vector, threshold):
        """
        information_gain method
        :param y_trn: {array-like}
        :param x_vector: {array-like}
        :param threshold: {array-like}
        :return: {float}
        """
        # Split the node to left child and right child
        left_indices, right_indices = self.split_node(x_vector, threshold)
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        n_labels = len(y_trn)

        # Calculate the parent entropy
        parent = entropy(y_trn)

        # Get the number of labels of the left child and right child
        n_left, n_right = len(left_indices), len(right_indices)

        # Calculate the left children entropy
        l_child = entropy(y_trn[left_indices])

        # Calculate the right children entropy
        r_child = entropy(y_trn[right_indices])

        # Calculate the weighted average of the entropy of the children
        children = (n_left / n_labels) * l_child \
                   + (n_right / n_labels) * r_child

        # Calculate information gain - difference in loss
        info_gain = parent - children

        return info_gain

    def split_node(self, x_vector, threshold):
        """
        split_node method
        :param x_vector: {array-like}
        :param threshold: {int}
        :return: {array-like}, {array-like}
        """
        # Find the non-zero grouped elements of the left node indices if the vector column
        # is less or equal the threshold
        left_indices = np.argwhere(x_vector <= threshold).flatten()

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
