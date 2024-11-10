import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression


# Dataset
X, y = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=7, random_state=51)

# Split the dataset into testing and training portions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# Regression model
clf = LogisticRegression(lr=0.001, max_iter=2000)

# Fit the model with the training data
clf.fit(X_train,y_train)

# Model coefficients
print(clf.coef_)
print(clf.intercept_)

# Make predictions
predictions = clf.predict(X_test)

# Accuracy of the model
acc = np.sum(predictions==y_test) / len(y_test)
print(acc)

# Plot the results
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'#FCE3BD', 1:'#D4B4EA'}
fig, ax = plt.subplots()
classes = df.groupby('label')
for key, clss in classes:
    clss.plot(ax=ax, kind='scatter', x='x', y='y', color=colors[key])

def f(x):
    return (-clf.intercept_ - clf.coef_[0] * x) / clf.coef_[1]

x = np.array(range(-24,24))  
plt.plot(x, f(x), color='#514559')  

plt.show()
