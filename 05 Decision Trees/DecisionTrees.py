import numpy as np
import pandas as pd
from collections import Counter
import math


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    """
    A decision tree classifier.

    Parameters
    ----------
    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split. Supported criteria are 'gini'
        for the Gini Impurity and 'entropy' for the Kullback–Leibler divergence.

    splitter : {'best', 'random'}, default='best'
        The strategy used to choose the split at each node. Supported strategies are
        'best' and 'random' to choose the best split and “random” to choose the best
        random split.

    max_depth : int, default=None
        The maximum depth of the tree.
        
        If None, the construction of the tree continues until either all the nodes
        are pure or all leaves have less than min_samples_split samples.
        
    min_samples_split : int or float, default=2
        The minimum number of samples required to allow further splitting of an 
        internal node.

        If float, 'ceil' function is applied to determine the min_samples_split.

    max_features : int, default=None
        The number of features to consider when looking for the best split.
        
    Attributes
    ----------
    max_features : int
        The number of max_features. It is either provided by the ised or inferred
        by the algorithm internally.
    """
    def __init__(self,*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, max_features=None):
        self._criterion = criterion
        self._splitter = splitter
        self._max_depth = max_depth
        self._min_samples_split = math.ceil(min_samples_split)
        self.max_features = max_features
        self._root = None

    def fit(self, X, y):
        """
        Build a decision tree classifier based on the training set provided.

        Parameters
        ----------
        X : {numpy.ndarray, pandas.Dataframe} of shape (n_samples, n_features)
            Training data.

        y : {numpy.ndarray, pandas.Dataframe} of shape (n_samples,)
            Target values as integers or strings.

        Returns
        -------
        self
            Fitted classifier.
        """
        if self._criterion not in ('gini', 'entropy'):
            raise ValueError("The 'criterion' parameter must be a str among {'gini', 'entropy'}.")
        if self._splitter not in ('best', 'random'):
            raise ValueError("The 'splitter' parameter must be a str among {'best', 'random'}.")
        if self._min_samples_split < 1:
            raise ValueError("The 'min_samples_split' patameter has to be greater than 1.")
        if self._max_depth is not None:
            if not isinstance(self._max_depth, int) or self._max_depth < 1:
                raise ValueError("The 'max_depth' parameter must be a positive natural number.")
        if self.max_features is not None:
            if not isinstance(self.max_features, int) or self.max_features < 1:
                raise ValueError("The 'max_features' parameter must be a positive natural number.")
        
        X = X.to_numpy() if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) else X
        y = y.to_numpy().flatten() if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series) else y

        self.max_features = X.shape[1] if not self.max_features else min(X.shape[1], self.max_features)
        self._root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # recursion stop criteria
        if (self._max_depth is not None and depth >= self._max_depth or n_labels == 1 or n_samples < self._min_samples_split):
            leaf_value = self._most_common_value(y)
            return Node(value=leaf_value)

        if self._splitter == 'random':
            feature_idxs = np.random.choice(n_features, self.max_features, replace=False)
        elif self._splitter == 'best':
            feature_idxs = np.array([x for x in range(n_features)])

        best_feature, best_threshold = self._best_split(X, y, feature_idxs)

        # child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_split_quality = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            col = X[:, feat_idx]
            thresholds = self._get_thresholds(col)

            for th in thresholds:
                split_quality = self._split_quality(y, col, th)
                if split_quality > best_split_quality:
                    best_split_quality = split_quality
                    split_idx = feat_idx
                    split_threshold = th

        return split_idx, split_threshold

    def _get_thresholds(self, col):
        col_sorted = sorted(col)
        thresholds = np.array([])
        for idx in range(len(col_sorted)-1):
            thresholds = np.append(thresholds, (col_sorted[idx] + col_sorted[idx+1]) / 2)
        return thresholds

    def _split_quality(self, y, col, threshold):
        left_idxs, right_idxs = self._split(col, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # weights for each leaf
        n_samples = len(y)
        weighted_l, weighted_r = len(left_idxs) / n_samples, len(right_idxs) / n_samples

        if self._criterion == 'gini':
            branch_gini = self._gini_index(y)

            gini_l, gini_r = self._gini_index(y[left_idxs]), self._gini_index(y[right_idxs])
            leaves_gini = (weighted_l) * gini_l + (weighted_r) * gini_r

            gini_impurity = branch_gini - leaves_gini
            return gini_impurity

        elif self._criterion == 'entropy':
            branch_entropy = self._entropy(y)

            entropy_l, entropy_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
            leaves_entropy = (weighted_l) * entropy_l + (weighted_r) * entropy_r

            information_gain = branch_entropy - leaves_entropy
            return information_gain

    def _split(self, col, threshold):
        left_idxs = np.argwhere(col <= threshold).flatten()
        right_idxs = np.argwhere(col > threshold).flatten()

        return left_idxs, right_idxs
    
    def _gini_index(self, y):
        value_counts = Counter(y).items()
        values_histogram = np.array([])
        for _, c in value_counts:
            values_histogram = np.append(values_histogram , c)
        probabilities = values_histogram / len(y)

        return 1 - np.sum(probabilities ** 2)

    def _entropy(self, y):
        value_counts = Counter(y).items()
        values_histogram = np.array([])
        for _, c in value_counts:
            values_histogram = np.append(values_histogram , c)
        probabilities = values_histogram / len(y)

        return -np.sum([p * np.log(p) for p in probabilities if p > 0])

    def _most_common_value(self, y):
        counts = Counter(y)
        return counts.most_common(1)[0][0]

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : {numpy.ndarray, pandas.Dataframe} of shape (n_features)
            Test samples.

        Returns
        -------
        y : numpy.ndarray of shape (n_queries,)
            Class labels for each data sample.
        """
        X = X.to_numpy() if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) else X
        return np.array([self._traverse_tree(x, self._root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def get_params(self):
        """
        Get parameters for the classifier.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {'criterion': self._criterion,
                'splitter': self._splitter,
                'max_depth': self._max_depth,
                'min_samples_split': self._min_samples_split,
                'max_features': self.max_features}

    def set_params(self,*,criterion='gini', splitter='best', max_depth=None, min_samples_split=2, max_features=None):
        """
        Set the parameters for the classifier.

        If parameters are not specified, the function sets them to
        default values.
        """
        self._criterion = criterion
        self._splitter = splitter
        self._max_depth = max_depth
        self._min_samples_split = math.ceil(min_samples_split)
        self.max_features = max_features
