import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    "age": [18, 25, 30, 35, 40],
    "salary": [20000, 35000, 40000, 60000, 80000]
}

df = pd.DataFrame(data)

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print("Scaled Data:")
print(scaled_data)
