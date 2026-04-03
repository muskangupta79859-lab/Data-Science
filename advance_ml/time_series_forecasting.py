"""
Simple Time Series Forecasting using Moving Average
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_data():
    dates = pd.date_range(start="2023-01-01", periods=100)
    values = np.random.randn(100).cumsum()
    
    df = pd.DataFrame({"Date": dates, "Value": values})
    df.set_index("Date", inplace=True)
    return df


def forecast(df, window=5):
    df["Moving_Avg"] = df["Value"].rolling(window=window).mean()
    
    # Forecast next value
    forecast_value = df["Value"].tail(window).mean()
    print(f"Next predicted value: {forecast_value:.2f}")
    
    return df


def plot(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["Value"], label="Original")
    plt.plot(df["Moving_Avg"], label="Moving Avg", linestyle="--")
    
    plt.legend()
    plt.title("Time Series Forecasting")
    plt.show()


def main():
    df = create_data()
    df = forecast(df)
    plot(df)


if __name__ == "__main__":
    main()
