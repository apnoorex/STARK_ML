import numpy as np
import pandas as pd


class KNeighborsRegressor:
    """
    Regression based on K nearest neighbors.

    The predicted value is determined as the average of the values of K nearest
    neighbors of the sample in the training set.

    Parameters
    ----------
    n_neighbors : int, default=3
        Number of neighbors to use by default.

    weights : {'uniform', 'distance'}, callable or None, default='uniform'
        Weight function used in prediction. Possible values:
        - 'uniform' : All points in each neighborhood
        are weighted equally.
        - 'distance' : Closer neighbors will have a greater influence than
        neighbors that are further away.

    Uniform weights are used by default. Standard Euclidean distance is used
    for both options.

    .. warning::

    If two neighbors have the same distances but different labels,
    the results will depend on the ordering of the training data.
    
    For more information about the KNN algorithm please visit:

    https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

    """
    def __init__(self, n_neighbors=3, weights='uniform'):
        self._k = n_neighbors
        self._weights_method = weights
        self._n_samples = 0

    def fit(self, X, y):
        """
        Fit the K nearest neighbors regressor from the training dataset.

        Parameters
        ----------
        X : {numpy.ndarray, pandas.Dataframe} of shape (n_samples, n_features)
            Training data.

        y : {numpy.ndarray, pandas.Dataframe} of shape (n_samples,)
            Target values.

        Returns
        -------
        self : KNeighborsRegressor
            Fitted K nearest neighbors regressor.
        """
        if len(X) < self._k:
            raise ValueError("The number of samples cannot be smaller than parameter 'K'")
        if len(X) != len(y):
            raise ValueError("'X' and 'y' are not of the same length")
        if self._weights_method not in ('uniform', 'distance'):
            raise ValueError("The 'weights' parameter must be a str among {'uniform', 'distance'}")

        self._n_samples = len(X)
        self.X_train = X.to_numpy() if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) else X
        self.y_train = y.to_numpy() if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series) else y

    def predict(self, X):
        """
        Predict the target for the provided data.

        Parameters
        ----------
        X : {numpy.ndarray, pandas.Dataframe} of shape (n_features)
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,)
            Target values.
        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()
        predictions = [float(self._predict(x)) for x in X]

        return np.array(predictions)

    def _predict(self, x):
        indices, dists = self._kneighbors(x, self._k, return_distance=True)
        labels = [self.y_train[i] for i in indices]

        if self._weights_method == 'uniform':
            mean = np.mean(labels)

        elif self._weights_method == 'distance':
            weights = self._get_weights(dists)

            labels_weighted = []
            for i in range(self._k):
                count = weights[i]
                while count > 0:
                    labels_weighted.append(labels[i])
                    count -= 1
            mean = np.mean(labels_weighted)

        return mean
    
    def _kneighbors(self, x, n_neighbors, return_distance=True):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        indices = np.argsort(distances)[:n_neighbors]
        dists = np.array([float(distances[i]) for i in indices])

        if return_distance:
            return indices, dists
        return indices

    def _get_weights(self, data):
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        interval_bins = [data_norm < 0.5, data_norm < 0.75, data_norm >= 0.75]
        weight_values = [5, 3, 1]
        weights = np.select(interval_bins, weight_values)

        return weights

    def _euclidean_distance(self, x1, x2):
        distance = np.sqrt(np.sum((x1 - x2) ** 2))

        return distance

    def kneighbors(self, X, n_neighbors, return_distance=True):
        """
        Find the K nearest neighbors of a point or points.

        Returns indices of and distances to the K specified neighbors
        of a point or points. A test set needs to be fitted first (See
        the '.fit()' method).

        Parameters
        ----------
        X : {numpy.ndarray, pandas.Dataframe}, shape (n_queries, n_features)
            Coordinates of points

        n_neighbors : int, default=3
            Number of neighbors to return.

        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neighbors_dist : numpy.ndarray of shape (n_queries, n_neighbors)
            An array containing the distances to points, only present if
            parameter return_distance is set to 'True'.

        neighbors_idx : ndarray of shape (n_queries, n_neighbors)
            Indices of the K nearest points in the training set.
        """
        if n_neighbors > self._n_samples:
            raise ValueError("The number neighbors provided is greater than the number of samples in the training set")
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()

        if return_distance:
            neighbors_idx, neighbors_dist = [], []
            for x in X:
                neigh, dist = self._kneighbors(x, n_neighbors, True)
                neighbors_idx.append(neigh)
                neighbors_dist.append(dist)
            return neighbors_dist, neighbors_idx

        neighbors_idx = []
        for x in X:
            neigh = self._kneighbors(x, n_neighbors, False)
            neighbors_idx.append(neigh)
        return neighbors_idx

    def get_params(self) -> dict:
        """
        Get parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {'n_neighbors': self._k, 'weights': self._weights_method}

    def set_params(self, n_neighbors=3, weights='uniform'):
        """
        Set the parameters of this estimator.

        If the parameters are not specified, the function sets the parameters to
        default values: n_neighbors = 3, weights = 'uniform'.
        """
        self._k = n_neighbors
        self._weights_method = weights
