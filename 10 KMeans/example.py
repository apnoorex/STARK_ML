import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from KMeans import KMeans

# Create dataset
X, y = make_blobs(centers=3, n_samples=45, n_features=2, shuffle=True, random_state=51)

# KMeans clustering model
k = KMeans(n_clusters=3, max_iter=150, random_state=17)

# Fit the model and make predictions
k.fit_predict(X)

# Print the coordinates of cluster centers
print('Cluster centers:', k.cluster_centers_)

# Plot the results
fig, ax = plt.subplots(figsize=(8, 6))

colors = ['#FEB6B9', '#BBDED7', '#8AC6D1']

for idx, cluster in enumerate(k._clusters):
    points = X[cluster].T
    ax.scatter(*points, c=colors[idx])

for point in k.cluster_centers_:
    ax.scatter(*point, marker="x", color="black", linewidth=2)

plt.show()
