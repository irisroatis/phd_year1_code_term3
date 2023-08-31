import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# Generate synthetic data
n_samples = 300
n_clusters = 3

X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

# Fit K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# Fit Logistic Regression for classification
clf = LogisticRegression(random_state=42)
clf.fit(X, y)
y_pred = clf.predict(X)

# Get cluster boundaries (using Voronoi diagram) for K-Means
h = 0.02  # Step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z_kmeans = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z_kmeans = Z_kmeans.reshape(xx.shape)

# Get decision boundaries for Logistic Regression
Z_logreg = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_logreg = Z_logreg.reshape(xx.shape)

# Create a figure with subplots

plt.scatter(X[:, 0], X[:, 1], color = 'grey')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

plt.scatter(X[:, 0], X[:, 1], c = y_pred)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Unsupervised Clustering (K-Means)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)
plt.contourf(xx, yy, Z_kmeans, alpha=0.2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Supervised Classification (Logistic Regression)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.contourf(xx, yy, Z_logreg, alpha=0.2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

