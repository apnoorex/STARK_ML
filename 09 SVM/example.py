import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from SVC import SVC


# Dataset
X, y = datasets.make_circles(n_samples=200, factor=.5, noise=.1, random_state=14)

# Values for the classes are changed to 1 and -1
y = np.where(y <= 0, -1, 1)

# Split the dataset into testing and training portions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

# Classification model
clf = SVC(C=1.2, degree=2)

# Fit the model with the training data
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Accuracy of the model
acc = np.sum(y_test == predictions) / len(y_test)
print(acc)

# Plot the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='Dark2')
plt.show()
