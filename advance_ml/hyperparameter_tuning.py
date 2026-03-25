"""
Hyperparameter Tuning using GridSearchCV
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load data
data = load_iris()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()

# Parameters
params = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10]
}

# Grid Search
grid = GridSearchCV(model, params, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)
