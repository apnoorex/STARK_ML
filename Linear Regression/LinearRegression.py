import pandas as pd
import numpy as np


class LinearRegression:
    """
    Linear Regression model based on the Ordinary Least Squares algorithm.

    The regression fits a linear model by calculating the coefficients 
    w1, w2, ..., wp that minimize the residual sum of squares between
    the observed targets in the training dataset. That linear model is 
    then used to make predictions
    based on these coefficients.

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.

    intercept_ : float
        Y intercept of the linear model.
        
    Note
    ----
    The idea behind the OLS algorithm used is explained in this article:
    
    https://blog.dailydoseofds.com/p/why-sklearns-linear-regression-has

    """
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._beta = None
        self._is_df = None

    def fit(self, X, y):
        """
        Fit linear model.
        
        Parameters
        ----------
        X : {numpy.ndarray, pandas.Dataframe} of shape (n_samples, n_features)
            Training data.
        
        y : {numpy.ndarray, pandas.Dataframe} of shape (n_samples,) or (n_samples, 1)
            Target values. The model can only handle one target.
                
        Returns
        -------
        self : object
            Fitted Estimator.
        """        
        self._is_df = isinstance(X, pd.DataFrame)

        if self._is_df:
            X = X.to_numpy()
            y = y.to_numpy()

        X = np.insert(X, 0, 1, axis=1)

        self._beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y).flatten()
        self.intercept_ = self._beta[0]
        self.coef_ = self._beta[1:]

    def predict(self, X):
        """
        Make predictions using the linear model.
        
        Parameters
        ----------
        X : {numpy.ndarray, pandas.Dataframe} of shape (n_samples, n_features)
            Samples to calculate predictions for.
        
        Returns
        -------
        C : numpy.ndarray, shape (n_samples,)
            Array of predicted values. The output for the convenience is in the
            form [[] [] ... []] if the data to the .fit() method was provided in
            pandas.Dataframe format and is flat otherwise.
        """        
        X = np.insert(X, 0, 1, axis=1)

        preds = np.sum(np.multiply(X, self._beta), axis=1)

        if self._is_df:
            return np.array([[pred] for pred in preds])
        return preds
