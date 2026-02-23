from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Model
model = RandomForestClassifier(random_state=42)

# Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross validation score
scores = cross_val_score(model, X, y, cv=cv)

print("Cross Validation Scores:", scores)
print("Average Accuracy:", np.mean(scores))
