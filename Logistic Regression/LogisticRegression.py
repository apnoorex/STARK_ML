import numpy as np


def sigmoid(y):
    return 1 / (1 + np.exp(-y))

def progress(coef_prev, inpt_prev, coef, inpt):
    return (np.sum((coef_prev - coef) ** 2) + (inpt_prev - inpt) ** 2)

class LogisticRegression():
    """
    Logistic Regression classifier based on Gradient Descent approach.

    Parameters
    ----------
    lr : float, default=0.001
        Learning rate.

    max_iter : int, default=1000
        Maximum number of iterations for the Gradient Descent to take. The algorithm
        can stop early if the change in weights and bias going from one iteration to
        the next iteration is too small.

    tol : float, default=10^-5*lr
        Tolerance for early stopping criteria.
        
    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients of the decision function.

    intercept_ : float
        Y intercept (bias) of the decision function.
    """
    def __init__(self, lr=0.001, max_iter=1000, tol=None):
        self._lr = lr
        self._max_iter = max_iter
        self._tol = pow(10, -5) * self._lr if tol is None else tol
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit the Logistic Regression model.

        Parameters
        ----------
        X : {numpy.ndarray, pandas.Dataframe} of shape (n_samples, n_features)
            Training data.

        y : {numpy.ndarray, pandas.Dataframe} of shape (n_samples,)
            Target values.

        Returns
        -------
        self
            Fitted classifier.
        """
        if self._lr >= 1:
            raise ValueError("Learning rate has to be less than 1")
        
        n_samples, n_features = X.shape
        self.coef_ = np.ones(n_features)
        self.intercept_ = 1

        for _ in range(self._max_iter):
            linear_prediction = np.dot(X, self.coef_) + self.intercept_
            predictions = sigmoid(linear_prediction)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            coef_prev = self.coef_
            inpt_prev = self.intercept_

            self.coef_ = self.coef_ - self._lr * dw
            self.intercept_ = self.intercept_ - self._lr * db

            if progress(coef_prev, inpt_prev, self.coef_, self.intercept_) < self._tol:
                break

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : {numpy.ndarray, pandas.Dataframe} of shape (n_features)
            Data samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Array containing class label for each data sample.
        """
        linear_prediction = np.dot(X, self.coef_) + self.intercept_
        predictions = sigmoid(linear_prediction)
        class_pred = [0 if y <= 0.5 else 1 for y in predictions]

        return np.array(class_pred)

    def get_params(self):
        """
        Get parameters for this classifier.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {'lr': self._lr, 'max_iter': self._max_iter, 'tol': self._tol}

    def set_params(self, lr=0.001, max_iter=1000, tol=None):
        """
        Set the parameters of this classifier.

        If the parameters are not specified, the function sets the parameters to
        default values: lr=0.001, max_iter=1000, tol=pow(10, -5)*lr.
        """
        self._lr = lr
        self._max_iter = max_iter
        self._tol = pow(10, -5) * self._lr if tol is None else tol
