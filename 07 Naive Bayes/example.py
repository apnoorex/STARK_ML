import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from GB1 import GaussianNB


# Dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Split the dataset into testing and training portions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

# Classification model
clf = GaussianNB(var_smoothing=1e-3)

# Fit the model with the training data
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Accuracy of the model
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)
