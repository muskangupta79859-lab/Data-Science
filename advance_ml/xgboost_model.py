"""
XGBoost Model Implementation

- Load dataset (Iris)
- Train-test split
- Train XGBoost model
- Evaluate accuracy
- Plot feature importance
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def load_data():
    """Load Iris dataset"""
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y


def split_data(X, y):
    """Split dataset into train and test"""
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    """Train XGBoost model"""
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    return accuracy


def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    importance = model.feature_importances_

    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importance)
    plt.xlabel("Importance Score")
    plt.title("Feature Importance - XGBoost")
    plt.tight_layout()
    plt.show()


def main():
    # Load data
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Plot feature importance
    plot_feature_importance(model, X.columns)


if __name__ == "__main__":
    main()
