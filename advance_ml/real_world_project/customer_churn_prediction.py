"""
Customer Churn Prediction Project
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data():
    # sample dataset (you can replace later with real CSV)
    data = {
        "Age": [25, 40, 35, 28, 50],
        "Salary": [50000, 80000, 60000, 52000, 90000],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "Churn": [0, 1, 0, 0, 1]
    }
    return pd.DataFrame(data)


def preprocess(df):
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    return df


def main():
    df = load_data()
    df = preprocess(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))


if __name__ == "__main__":
    main()
