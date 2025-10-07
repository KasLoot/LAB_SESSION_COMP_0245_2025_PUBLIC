import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor

# Data dictionary containing the dataset
data = {
    'age': [23, 31, 35, 35, 42, 43, 45, 46, 46, 51],
    'likes goats': [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    'likes height': [0, 1, 1, 0, 0, 1, 0, 1, 1, 1],
    'climbed meters': [200, 700, 600, 300, 200, 700, 300, 700, 600, 700]
}

# Creating DataFrame from the data dictionary
df = pd.DataFrame(data)

# Defining the feature matrix (X) and the target vector (y)
X = df[['age', 'likes goats', 'likes height']].values
y = df['climbed meters'].values

# Creating the Decision Tree Regressor model called tree
tree = DecisionTreeRegressor()
tree.fit(X=X, y=y)

# Plotting the decision tree
plt.figure(figsize=(16, 10))  # Set the size of the figure
plot_tree(tree, feature_names=['age', 'likes goats', 'likes height'], fontsize=6)
plt.show()  # Show the plot