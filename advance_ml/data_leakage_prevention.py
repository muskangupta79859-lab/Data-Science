"""
Data Leakage Prevention using Pipeline
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def load_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y


def main():
    X, y = load_data()

    # Split BEFORE scaling (important to avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline (preprocessing + model)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier())
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy without leakage: {accuracy:.4f}")


if __name__ == "__main__":
    main()
