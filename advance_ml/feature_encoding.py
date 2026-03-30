"""
Feature Encoding Techniques
- Label Encoding
- One-Hot Encoding
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data():
    data = {
        "Name": ["Aman", "Sara", "John", "Riya"],
        "City": ["Delhi", "Mumbai", "Delhi", "Bangalore"],
        "Gender": ["Male", "Female", "Male", "Female"]
    }
    return pd.DataFrame(data)


def label_encoding(df):
    le = LabelEncoder()
    df["Gender_Encoded"] = le.fit_transform(df["Gender"])
    return df


def one_hot_encoding(df):
    df_encoded = pd.get_dummies(df, columns=["City"])
    return df_encoded


def main():
    df = load_data()
    print("Original Data:\n", df)

    df = label_encoding(df)
    print("\nAfter Label Encoding:\n", df)

    df = one_hot_encoding(df)
    print("\nAfter One-Hot Encoding:\n", df)


if __name__ == "__main__":
    main()
