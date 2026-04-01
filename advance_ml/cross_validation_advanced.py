"""
Advanced Cross Validation Techniques
- KFold
- StratifiedKFold
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier


def load_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y


def kfold_cv(X, y):
    model = RandomForestClassifier()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=kf)
    print("KFold Scores:", scores)
    print("KFold Mean Accuracy:", scores.mean())


def stratified_kfold_cv(X, y):
    model = RandomForestClassifier()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=skf)
    print("\nStratifiedKFold Scores:", scores)
    print("StratifiedKFold Mean Accuracy:", scores.mean())


def main():
    X, y = load_data()

    kfold_cv(X, y)
    stratified_kfold_cv(X, y)


if __name__ == "__main__":
    main()
