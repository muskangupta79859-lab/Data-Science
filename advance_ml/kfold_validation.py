"""
K-Fold Cross Validation Example
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X, y = data.data, data.target

# KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier()

scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = accuracy_score(y_test, y_pred)
    scores.append(score)

print("Scores:", scores)
print("Average Accuracy:", sum(scores)/len(scores))
