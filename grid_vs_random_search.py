from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Dataset
data = load_iris()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)

# Parameter space
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}

# GRID SEARCH
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Grid Params:", grid.best_params_)
print("Grid Accuracy:", grid.score(X_test, y_test))

# RANDOM SEARCH
random = RandomizedSearchCV(
    model,
    param_distributions=param_grid,
    n_iter=5,
    cv=5,
    random_state=42
)
random.fit(X_train, y_train)

print("Best Random Params:", random.best_params_)
print("Random Accuracy:", random.score(X_test, y_test))
