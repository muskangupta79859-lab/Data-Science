import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    "hours_studied": [1,2,3,4,5],
    "sleep_hours": [5,6,7,6,5]
}

df = pd.DataFrame(data)

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

print("Original Data")
print(df)

print("\nNormalized Data")
print(scaled_df)
