"""
Feature Importance using RandomForest
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def load_data():
    return pd.read_csv("data/churn.csv")


def preprocess(df):
    df = df.dropna()
    
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    return df


def main():
    df = load_data()
    df = preprocess(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    features = X.columns

    # Plot
    plt.barh(features, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.show()


if __name__ == "__main__":
    main()
