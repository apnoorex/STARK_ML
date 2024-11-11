import numpy as np
import pandas as pd
from collections import Counter
import math
from DecisionTrees import DecisionTreeClassifier


class RandomForestClassifier:
    """
    A random forest classifier.

    A random forest is an estimator that combines many decision trees by averaging
    their predictions to improve the accuracy and avoid overfitting.

    Trees in the forest use the 'random' split strategy, i.e. equivalent to passing
    'splitter='random'' to the underlying class DecisionTreeClassifier.

    The sub-sample size is determined by the 'max_samples' parameter if 
    'bootstrap=True' (default), otherwise the whole dataset is used to build each
    tree.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split when building each tree.
        Supported criteria are 'gini' for the Gini Impurity and 'entropy' for
        the Kullback–Leibler divergence.

    max_depth : int, default=None
        The maximum depth of each tree in the forest.
        
        If None, the construction of the tree continues until either all the nodes
        are pure or all leaves have less than min_samples_split samples.
        
    min_samples_split : int or float, default=2
        The minimum number of samples required to allow further splitting of an 
        internal node of each tree in the forest.

        If float, 'ceil' function is applied to determine the min_samples_split.

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split.

        If int, then consider 'max_features' features ar each split.
        If float, then 'ceil' function is applied to determine the number of
        features.
        If 'sqrt', then `max_features=sqrt(n_features)`.
        If 'log2', then `max_features=log2(n_features)`.

    bootstrap : bool, default=True
        Whether or not to bootstrap samples. If False, the whole dataset is used
        for each tree.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from the dataset to
        build each tree.

        If None (default), then draw 'X.shape[0]' samples.
        If int, then draw `max_samples` samples.
        If float, then draw `max(round(n_samples * max_samples), 1)` samples. In
        this case `max_samples` should be in the interval `(0.0, 1.0]`.
    """
    def __init__(
            self,
            n_estimators=100, *,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            max_features='sqrt',
            bootstrap=True,
            max_samples=None):
        # Parameters
        self._n_estimators = n_estimators
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._max_features = max_features
        self._bootstrap = bootstrap
        self._max_samples = max_samples
        # Variables
        self._n_samples = 0
        self._trees = []

    def fit(self, X, y):
        """
        Build a random forest based on the training set provided.

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
        if self._n_estimators < 1:
            raise ValueError("Parameter 'n_estimators' must be an int greater than 0.")
        if self._criterion not in ('gini', 'entropy'):
            raise ValueError("Parameter 'criterion' must be a str among {'gini', 'entropy'}.")
        if self._min_samples_split < 1:
            raise ValueError("Parameter 'min_samples_split' must be greater than 1.")
        if self._max_depth is not None:
            if not isinstance(self._max_depth, int) or self._max_depth < 1:
                raise ValueError("Parameter 'max_depth' must be a positive natural number.")
        if ((isinstance(self._max_features, str) and self._max_features not in ('sqrt', 'log2')) or
            (isinstance(self._max_features, int) and self._max_features < 1)):
            raise ValueError("Parameter 'max_samples' must be '{''sqrt', 'log2', None'}' or int greater than 0.")

        self._n_samples = X.shape[0]
        self._max_features = self._get_max_features()

        X = X.to_numpy() if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) else X
        y = y.to_numpy().flatten() if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series) else y

        for _ in range(self._n_estimators):
            tree = DecisionTreeClassifier(
                criterion = self._criterion,
                splitter = 'random',
                max_depth = self._max_depth,
                min_samples_split = self._min_samples_split,
                max_features = self._max_features)
            
            if self._bootstrap:
                X_train, y_train = self._bootstrap_samples(X, y)
                tree.fit(X_train, y_train)
            else:
                tree.fit(X, y)

            self._trees.append(tree)

    def _get_max_features(self):
        if self._max_features == 'sqrt':
            return math.ceil(math.sqrt(self._n_samples))
        elif self._max_features == 'log2':
            return math.ceil(math.log2(self._n_samples))
        elif self._max_features is None:
            return self._n_samples
        else:
            return math.ceil(self._n_samples)

    def _bootstrap_samples(self, X, y):
        self._max_samples = self._get_max_samples()
        idxs = np.random.choice(self._n_samples, self._max_samples, replace=True)
        return X[idxs], y[idxs]
    
    def _get_max_samples(self):
        if self._max_samples is None:
            return self._n_samples
        elif isinstance(self._max_samples, int):
            if self._max_samples < 1:
                raise ValueError("Wrong value for the 'max_samples' parameter. Please see help for details.")
            return self._max_samples
        elif isinstance(self._max_samples, float):
            if self._max_samples < 0.0 or self._max_samples > 1.0:
                raise ValueError("Wrong value for the 'max_samples' parameter. Please see help for details.")
            return max(round(self._n_samples * self._max_samples), 1)
        else:
            raise ValueError("Wrong value for the 'max_samples' parameter. Please see help for details.")

    def _most_common_value(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """
        Predict the class labels for the provided data.
        
        The predicted class
        is determined as a majority vote by the trees in the forest.

        Parameters
        ----------
        X : {numpy.ndarray, pandas.Dataframe} of shape (n_features)
            Test samples.

        Returns
        -------
        y : numpy.ndarray of shape (n_queries,)
            Class labels for each data sample.
        """
        predictions = [tree.predict(X) for tree in self._trees]
        tree_predictions = list(map(list, zip(*predictions)))
        predictions = [self._most_common_value(pred) for pred in tree_predictions]

        return np.array(predictions)

    def get_params(self):
        """
        Get parameters for the classifier.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {'n_estimators': self._n_estimators,
                'criterion': self._criterion,
                'max_depth': self._max_depth,
                'min_samples_split': self._min_samples_split,
                'max_features': self._max_features,
                'bootstrap': self._bootstrap,
                'max_samples': self._max_samples}

    def set_params(
            self,
            n_estimators=100, *,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            max_features='sqrt',
            bootstrap=True,
            max_samples=None):
        """
        Set parameters for the classifier.

        If a parameter is not specified, the function sets it to
        default value.
        """
        self._n_estimators = n_estimators
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._max_features = max_features
        self._bootstrap = bootstrap
        self._max_samples = max_samples
