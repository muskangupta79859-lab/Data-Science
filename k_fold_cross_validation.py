import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_iris()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Model
model = RandomForestClassifier()

# K-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5)

print("Cross Validation Scores:", scores)
print("Average Score:", scores.mean())
