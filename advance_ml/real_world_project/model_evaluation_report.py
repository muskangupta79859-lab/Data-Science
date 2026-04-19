"""
Model Evaluation Report
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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

    preds = model.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))


if __name__ == "__main__":
    main()
