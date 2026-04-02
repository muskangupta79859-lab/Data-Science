"""
Time Series Basics - Trend Visualization & Moving Average
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_data():
    # dummy time series data
    dates = pd.date_range(start="2023-01-01", periods=100)
    values = np.random.randn(100).cumsum()
    
    df = pd.DataFrame({"Date": dates, "Value": values})
    df.set_index("Date", inplace=True)
    return df


def moving_average(df):
    df["Moving_Avg"] = df["Value"].rolling(window=5).mean()
    return df


def plot_data(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["Value"], label="Original")
    plt.plot(df["Moving_Avg"], label="Moving Average", linestyle="--")
    
    plt.legend()
    plt.title("Time Series with Moving Average")
    plt.show()


def main():
    df = create_data()
    df = moving_average(df)
    
    print(df.head())
    plot_data(df)


if __name__ == "__main__":
    main()
