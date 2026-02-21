import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Dataset load
data = load_iris()
X = data.data
y = data.target

# Model 1: Logistic Regression Pipeline
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=200))
])

# Model 2: Decision Tree
model_dt = DecisionTreeClassifier()

# Cross-validation scores
scores_lr = cross_val_score(pipe_lr, X, y, cv=5)
scores_dt = cross_val_score(model_dt, X, y, cv=5)

print("Logistic Regression CV Accuracy:", scores_lr.mean())
print("Decision Tree CV Accuracy:", scores_dt.mean())
