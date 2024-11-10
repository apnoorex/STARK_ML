import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from LogisticRegression import LogisticRegression


# Dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

# Split the dataset into testing and training portions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

# Regression model
clf = LogisticRegression(lr=0.001, max_iter=2000)

# Fit the model with the training data
clf.fit(X_train,y_train)

# Make predictions
predictions = clf.predict(X_test)

# Accuracy of the model
acc = np.sum(predictions==y_test) / len(y_test)
print(acc)
