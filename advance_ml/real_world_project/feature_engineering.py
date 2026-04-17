"""
Feature Engineering for Customer Churn
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data():
    return pd.read_csv("data/churn.csv")


def preprocess(df):
    df = df.dropna()

    # Encode categorical
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


def feature_engineering(df):
    # Example feature creation
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['TotalSpend'] = df['tenure'] * df['MonthlyCharges']

    return df


def scale_features(df):
    scaler = StandardScaler()
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def main():
    df = load_data()
    df = preprocess(df)
    df = feature_engineering(df)

    X, y = scale_features(df)

    print("Feature Engineering Done")
    print("Shape:", X.shape)


if __name__ == "__main__":
    main()
