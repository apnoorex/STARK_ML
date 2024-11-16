import numpy as np


class PCA:
    """
    Principal component analysis (PCA).

    The input data is centered, but not scaled.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep. If the parameter is not set, all components are
        kept.

    copy : bool, default=True
        If True, the data passed to fit is copied internally and is not getting 
        overwritten. This allows running fit(X).transform(X). If set to False,
        fit_transform(X) can be used instead.

    Attributes
    ----------
    components_ : numpy.ndarray of shape (n_components, n_features)
        The right singular vectors of the centered input data, parallel to its
        eigenvectors. The components are sorted in the descending order.

    mean_ : numpy.ndarray of shape (n_features,)
        Per-feature mean, estimated from the training set. Equal to 'X.mean(axis=0)'.
    """
    def __init__(self, n_components=None, *, copy=True):
        self._n_components_ = n_components
        self._copy = copy
        self.components_ = None
        self.mean_ = None

    def fit(self, X, y=None):
        """
        Fit the model with input data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Ignored.

        Returns
        -------
        self : PCA
            Fitted estimator.
        """
        if self._n_components_ is not None:
            if not isinstance(self._n_components_, int) or self._n_components_ < 1:
                raise ValueError("Parameter 'n_components' must be a positive natural number.")

        self._fit(X)
        
        return self

    def _fit(self, X):
        if self._n_components_ is None:
            self._n_components_ = X.shape[1]

        # Center data
        self.mean_ = np.mean(X, axis=0)

        if self._copy:
            X_copy = X - self.mean_
            cov = np.cov(X_copy.T)
        else:
            X = X - self.mean_
            cov = np.cov(X.T)

        eigenvectors, eigenvalues = np.linalg.eig(cov)
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idxs], eigenvectors[idxs]

        self.components_ = eigenvectors[:self._n_components_]

        if self._copy:
            return X_copy
        else:
            return X

    def transform(self, X):
        """
        Apply dimensionality reduction to the input data X.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_features, )
            New data.

        Returns
        -------
        y : numpy.ndarray of shape (n_queries,)
            Projection of X on the first principal components.
        """
        X = X - self.mean_
        return np.dot(X, self.components_.T)

    def fit_transform(self, X, y=None):
        """
        Fit the model with X and apply the dimensionality reduction on it.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_features)
            New data.

        y : Ignored
            Ignored.

        Returns
        -------
        y : numpy.ndarray of shape (n_queries,)
            Projection of X on the first principal components.
        """
        X = self._fit(X)
        return np.dot(X, self.components_.T)
    
    def get_params(self):
        """
        Get parameters for the estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {'n_components': self._n_components_, 'copy': self._copy}

    def set_params(self, n_components=None, *, copy=True):
        """
        Set parameters for the estimator.

        If parameters are not specified, the function sets them to
        default values.
        """
        self._n_components_ = n_components
        self._copy = copy
    