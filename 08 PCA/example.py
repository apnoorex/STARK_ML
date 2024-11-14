from PCA import PCA
import matplotlib.pyplot as plt
from sklearn import datasets

# Dataset
data = datasets.load_iris()
X, y = data.data, data.target

# PCA model
pca = PCA()

# Fit the model and transform X
X_projected = pca.fit_transform(X)

#Plot the results
pc1 = X_projected[:, 0]
pc2 = X_projected[:, 1]
plt.scatter(pc1, pc2, c=y, cmap='copper_r')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
