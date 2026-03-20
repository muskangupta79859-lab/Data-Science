import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = {
    "age": [20, 25, 30, 35, 40],
    "salary": [20000, 30000, 40000, 50000, 60000]
}

df = pd.DataFrame(data)

# Standard Scaling
std_scaler = StandardScaler()
std_scaled = std_scaler.fit_transform(df)

# Min-Max Scaling
mm_scaler = MinMaxScaler()
mm_scaled = mm_scaler.fit_transform(df)

print("Standard Scaled Data:")
print(std_scaled)

print("\nMin-Max Scaled Data:")
print(mm_scaled)
