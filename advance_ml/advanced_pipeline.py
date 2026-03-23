"""
Advanced ML Pipeline with Feature Selection + Cross Validation
Author: Muskan Gupta
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=chi2, k=2)),
    ('model', RandomForestClassifier())
])

# Cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5)

# Train
pipeline.fit(X_train, y_train)

# Test accuracy
test_score = pipeline.score(X_test, y_test)

print("Cross Validation Score:", scores.mean())
print("Test Accuracy:", test_score)
