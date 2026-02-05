import pandas as pd
from sklearn.preprocessing
import StandardScaler , MinMaxScaler

# Sample Dataset
data={
  "Age":[18,22,,25,30,,40] ,
  "Salary":[15000,20000,25000,30000,60000]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Standard Scaling
standard_scaler = StandardScaler()
df_standard = pd.DataFrame(standard_scaler.fit_transform(df),
                          columns = df.columns
                          )
print("\nStandard Scaled Data:")
print(df_standard)

# Min-Max Scaling
minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df),
                        columns = df.columns
                        )

print("\n Min-Max Scaled Data:")
print(df_minmax)
