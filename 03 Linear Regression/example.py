import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression


# Dataset
X, y = datasets.make_regression(n_samples=200, n_features=1, noise=20, random_state=51)

# # The Regression can accept either Numpy array or Dataframe as its input 
# X = pd.DataFrame(X, columns=['Feature1'])
# y = pd.DataFrame(y, columns=['Target'])

# Split the dataset into testing and training portions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

# Regression model
reg = LinearRegression()

# Fit the model with the training data
reg.fit(X_train, y_train)

# Model coefficients
print(reg.coef_)
print(reg.intercept_)

# Make predictions
predictions = reg.predict(X_test)

# Accuracy of the model
mse = np.mean((y_test - predictions) ** 2)
print(mse)

# Plot the results
predicted_line = reg.predict(X)
fig = plt.figure(figsize=(8,6))
m_train = plt.scatter(X_train, y_train, color=['#FCE3BD'], s=10)
m_test = plt.scatter(X_test, y_test, color=['#D4B4EA'], s=10)
plt.plot(X, predicted_line, color='#514559', linewidth=2)
plt.show()
