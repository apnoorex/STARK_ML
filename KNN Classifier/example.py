import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from KNeighborsClassifier import KNeighborsClassifier as KNN


# Dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the dataset into test and train portions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)

# Model parameters
n_neighbors = 17
weight = 'distance' # {'uniform', 'distance'}

# Classification model
clf = KNN(n_neighbors, weights=weight)

# Fit the model with the train data
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Accuracy of the model
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)
