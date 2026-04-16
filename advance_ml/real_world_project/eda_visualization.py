"""
EDA and Visualization for Customer Churn
"""

import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    df = pd.read_csv("data/churn.csv")
    return df


def basic_info(df):
    print("Dataset Info:\n")
    print(df.info())
    print("\nSummary:\n", df.describe())


def visualize(df):
    # Churn count
    df['Churn'].value_counts().plot(kind='bar')
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.show()

    # Example: Age distribution (if exists)
    if 'Age' in df.columns:
        df['Age'].hist()
        plt.title("Age Distribution")
        plt.show()


def main():
    df = load_data()
    
    basic_info(df)
    visualize(df)


if __name__ == "__main__":
    main()
