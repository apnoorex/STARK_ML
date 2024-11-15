import numpy as np


class SVC:
    """
    Support Vector Classification.

    Parameters
    ----------
    C : float, default=1.0
        Regulazation parameter. Must be strictly positive.

    degree : int, default=3
        Degree of polynomial kernel function. Must be >= 0.

    max_iter : int, default=1000
        Number of iteration for the solver.

    Note
    ----
    The idea behind the implementation of the algorithm was borrowed from here:

    https://adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-training-algorithms/
    """
    def __init__(self, *, C=1.0, degree=3, max_iter=1000):
        #Parameters
        self._C = C
        self._degree = degree
        self._max_iter = max_iter
        # Variables
        self._alpha = None
        self._b = None
        self._n_samples = None
        self._lr = 0.001
        self.X = None
        self.y = None

    def _kernel(self, A, B):
        return (1 + A.dot(B.T)) ** self._degree
        
    def fit(self, X, y):
        """
        Fit the SVC model with the given data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training data.

        y : numpy.ndarray of shape (n_samples,)
            Target values as  with 1 and -1 for class labels.

        Returns
        -------
        self : SVC
            Fitted classifier.
        """
        if self._C <= 0:
            raise ValueError("Parameter 'C' must be a positive float number")
        if self._degree < 1 or not isinstance(self._degree, int):
            raise ValueError("Parameter 'degree' must be a positive natural number")

        self.X = X
        self.y = y

        self._n_samples = X.shape[0]
        self._alpha = np.random.random(self._n_samples)

        kernel_matrix = np.outer(y, y) * self._kernel(X, X)

        for _ in range(self._max_iter):
            gradient = np.ones(self._n_samples) - kernel_matrix.dot(self._alpha)

            self._alpha += self._lr * gradient
            self._alpha[self._alpha > self._C] = self._C
            self._alpha[self._alpha < 0] = 0

        alpha_idxs = np.where((self._alpha) > 0 & (self._alpha < self._C))[0]
        
        # for b, only 0 < α < C are considered
        b_s = []        
        for idx in alpha_idxs:
            b_s.append(y[idx] - (self._alpha * y).dot(self._kernel(X, X[idx])))
        self._b = np.mean(b_s)

        return self
            
    def predict(self, X):
        """
        Predict the class labels for the provided data.
        
        Parameters
        ----------
        X : numpy.ndarray of shape (n_features)
            Test samples.

        Returns
        -------
        y : numpy.ndarray of shape (n_queries,)
            Class labels for each data sample.
        """
        approx = (self._alpha * self.y).dot(self._kernel(self.X, X)) + self._b
        return np.sign(approx)

    def get_params(self):
        """
        Get parameters for the classifier.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {'C': self._C, 'degree': self._degree, 'max_iter' : self._max_iter}

    def set_params(self, *, C=1.0, degree=3, max_iter=1000):
        """
        Set parameters for the classifier.

        If parameters are not specified, the function sets them to
        default values.
        """
        self._C = C
        self._degree = degree
        self._max_iter = max_iter
