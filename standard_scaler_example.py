import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    "age": [20, 25, 30, 35, 40],
    "salary": [20000, 30000, 40000, 50000, 60000]
}

df = pd.DataFrame(data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

print("Scaled Data:")
print(scaled_data)
