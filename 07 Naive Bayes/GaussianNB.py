import numpy as np
import pandas as pd


class GaussianNB:
    """
    Gaussian Naive Bayes classifier.

    Parameters
    ----------
    priors : {list or numpy.ndarray} of shape (n_classes,), default=None
        Prior probabilities of the classes.

    var_smoothing : float, default=1e-9
        Portion of the largest variance among all variance that is added
        to other variances to provide stability and take care of potential
        divide by zero errors.
    """
    def __init__(self, *, priors=None, var_smoothing=1e-09):
        self._priors = np.array(priors) if isinstance(priors, list) else priors
        self._var_smoothing = var_smoothing

    def fit(self, X, y):
        """
        Fit the classifier according to provided data.

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
        if self._var_smoothing <= 0:
            raise ValueError("Parameter 'var_smoothing' must be greater than 0.")

        X = X.to_numpy() if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) else X
        y = y.to_numpy().flatten() if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series) else y

        self._classes = np.unique(y)
        self._n_classes = len(self._classes)
        if self._priors is not None:
            if len(self._priors) != self._n_classes:
                raise ValueError("The number of prior probabilities provided doesn't match the number of classes in y.")

        n_samples, n_features = X.shape
        self._mean = np.zeros((self._n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((self._n_classes, n_features), dtype=np.float64)

        if self._priors is None:
            self._priors = np.zeros(self._n_classes, dtype=np.float64)
            for idx, clss in enumerate(self._classes):
                X_clss = X[y == clss]
                self._priors[idx] = X_clss.shape[0] / float(n_samples)

        for idx, clss in enumerate(self._classes):
            X_clss = X[y == clss]
            self._mean[idx, :] = X_clss.mean(axis=0)
            self._var[idx, :] = X_clss.var(axis=0)

        self._smoothen_variance()

    def _smoothen_variance(self):
        max_variance = np.max(self._var)
        self._var += max_variance * self._var_smoothing

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
        predictions = [self._predict(x) for x in X]

        return np.array(predictions)

    def _predict(self, x):
        posteriors = []
        for idx in range(self._n_classes):
            prior = np.log(self._priors[idx])
            likelihood = np.sum(np.log(self._likelihood(idx, x)))
            posterior = likelihood + prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _likelihood(self, class_idx, x):
        mu = self._mean[class_idx]
        sigma2 = self._var[class_idx]
        # Variables 'mu' and 'sigma2' are used for readability purposes
        likelihood = np.exp(-((x - mu) ** 2) / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)

        return likelihood

    def get_params(self):
        """
        Get parameters of the classifier.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {'priors': self._priors, 'var_smoothing': self._var_smoothing}

    def set_params(self, *, priors=None, var_smoothing=1e-09):
        """
        Set parameters for the classifier.

        If a parameter is not specified, the function sets it to
        default value.
        """
        self._priors = np.array(priors) if isinstance(priors, list) else priors
        self._var_smoothing = var_smoothing
