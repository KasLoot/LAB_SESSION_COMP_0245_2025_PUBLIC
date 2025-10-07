import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize decision tree classifiers

# Train the classifiers

# Predictions and evaluations


print("Accuracy using Gini index:", accuracy_score(y_test, y_pred_gini))
print("Accuracy using Entropy:", accuracy_score(y_test, y_pred_entropy))

# Plotting decision boundaries
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Function to plot decision boundaries
def plot_decision_boundaries(X, y, model, ax, title):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    ax.set_title(title)

plot_decision_boundaries(X_train, y_train, tree_gini, ax[0], 'Decision Tree (Gini)')
plot_decision_boundaries(X_train, y_train, tree_entropy, ax[1], 'Decision Tree (Entropy)')

plt.show()
