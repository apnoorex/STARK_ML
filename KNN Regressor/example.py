from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from KNeighborsRegressor import KNeighborsRegressor as KNN


# Dataset
df = data('turnout')
X = df[['age', 'income', 'vote']]
y = df['educate']

# Split the dataset into testing and training portions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)

# Model parameters
n_neighbors = 23
weight = 'distance' # {'uniform', 'distance'}

# Classification model
clf = KNN(n_neighbors, weights=weight)

# Fit the model with the training data
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Accuracy of the model
print(mse(y_test, predictions))
