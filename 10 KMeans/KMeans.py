import numpy as np
import pandas as pd
from numpy.random import RandomState


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    """
    K-Means clustering.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.

    max_iter : int, default=300
        Maximum number of iterations for the k-means algorithm to have in a single run.

    tol : float, default=0.0001
        Relative tolerance that determines the difference between the results of two
        consecutive iterations to declare convergence.

    random_state : int, default=None
        Determines random number generator used to initializes the centroids.
    """
    def __init__(self, n_clusters=8, *, max_iter=300, tol=0.0001, random_state=None):
        # Parameters
        self._K = n_clusters
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        # Variables
        self.X = None
        self._n_samples = None
        self.n_features_in_ = None
        self._clusters = [[] for _ in range(self._K)] # [[cl_1] [cl_2] ... [cl_n]]
        self.cluster_centers_ = [] # centers for each claster
        self._n_runs = 3 # number of runs to find the lowest variance

    def fit(self, X, y=None):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : {numpy.ndarray, pandas.Dataframe} of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Ignored.

        Returns
        -------
        self : KMeans
            Fitted estimator.
        """
        if self._K < 1 or not isinstance(self._K, int):
            raise ValueError("Parameter 'n_clusters' must be a positive natural number.")
        if self._max_iter < 1 or not isinstance(self._max_iter, int):
            raise ValueError("Parameter 'max_iter' must be a positive natural number.")
        if self._tol <= 0:
            raise ValueError("Parameter 'tol' must be a positive float number.")
        
        X = X.to_numpy() if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) else X

        self.X = X
        self._n_samples, self.n_features_in_ = X.shape

        rng_t = RandomState(self._random_state)
        random_states = rng_t.randint(100000, size=self._n_runs)

        # Variables to store the results for each run
        all_clusters, all_cluster_centers = [], []
        all_variances = np.array([])

        for run_idx in range(self._n_runs):
            # Randomly assign centroids
            rng = RandomState(random_states[run_idx])
            random_sample_idxs = rng.randint(self._n_samples, size=self._K)
            cluster_centers = [self.X[idx] for idx in random_sample_idxs]

            for _ in range(self._max_iter):
                clusters = self._create_clusters(cluster_centers)

                # New centroid for each cluster
                centers_prev = cluster_centers
                cluster_centers = self._new_centers(clusters)

                if self._progress(centers_prev, cluster_centers) < self._tol:
                    break

            total_variance = self._total_variance(clusters)

            all_variances = np.append(all_variances, total_variance)
            all_cluster_centers.append(cluster_centers)
            all_clusters.append(clusters)

        # Choose the best clustering based on the lowest variance
        best_run = np.argmin(all_variances)
        self._clusters = all_clusters[best_run]
        self.cluster_centers_ = all_cluster_centers[best_run]

        return self
    
    def _total_variance(self, clusters):
        """ Sum of variances in each cluster. """
        variances = [np.var(cluster) for cluster in clusters]
        return np.sum(variances)

    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : {numpy.ndarray, pandas.Dataframe} of shape (n_features)
            Data to predict.

        Returns
        -------
        y : numpy.ndarray of shape (n_queries,)
            Index of the cluster each sample in X belongs to.
        """
        X = X.to_numpy() if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) else X

        predictions = np.array([self._predict(x) for x in X], dtype=int)
        return predictions

    def _predict(self, x):
        return self._closest_centroid(x, self.cluster_centers_)
    
    def fit_predict(self, X, y=None):
        """
        Compute claster centers and predict the closest cluster for each sample in X.
        Equivalent of fit(X).predict(X).

        Parameters
        ----------
        X : {numpy.ndarray, pandas.Dataframe} of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Ignored.

        Returns
        -------
        y : numpy.ndarray of shape (n_queries,)
            Index of the cluster each sample in X belongs to.
        """
        X = X.to_numpy() if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) else X

        return self.fit(X).predict(X)

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self._K)]
        for idx, x in enumerate(self.X):
            centroid_idx = self._closest_centroid(x, centroids)
            clusters[centroid_idx].append(idx)
            
        return clusters

    def _closest_centroid(self, x, centroids):
        distances = [euclidean_distance(x, centroid) for centroid in centroids]
        closest_centroid_idx = np.argmin(distances)

        return closest_centroid_idx

    def _new_centers(self, clusters):
        """ New centers as the mean value of the cluster values """
        centroids = np.zeros((self._K, self.n_features_in_))
        for idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[idx] = cluster_mean

        return centroids

    def _progress(self, centers_prev, centers):
        improvements = []
        for idx in range(self._K):
            improvement = np.sum((centers_prev[idx] - centers[idx]) ** 2)
            improvements.append(improvement)

        return sum(improvements)
    
    def get_params(self):
        """
        Get parameters for the estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {'n_clusters': self._K,
                'max_iter': self._max_iter,
                'tol': self._tol,
                'random_state': self._random_state}

    def set_params(self, n_clusters=8, *, max_iter=300, tol=0.0001, random_state=None):
        """
        Set parameters for the estimator.

        If parameters are not specified, the function sets them to
        default values.
        """
        self._K = n_clusters
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
