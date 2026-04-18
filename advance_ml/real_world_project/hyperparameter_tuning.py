"""
Hyperparameter Tuning using GridSearchCV
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20]
    }

    grid = GridSearchCV(model, param_grid, cv=3)
    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best Score:", grid.best_score_)


if __name__ == "__main__":
    main()
