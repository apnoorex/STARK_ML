import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from RandomForest import RandomForestClassifier


# Dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Split the dataset into testing and training portions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

# Classification model
clf = RandomForestClassifier(n_estimators=20, max_depth=5, max_samples=50)

# Fit the model with the training data
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Accuracy of the model
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)
