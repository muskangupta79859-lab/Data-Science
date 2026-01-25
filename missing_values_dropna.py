import pandas as pd
import numpy as np
data={
  "Name":["Aman","Riya","Kunal","Neha"],
  "Age":[20,np.nan,22,np.nan],
  "Marks":[85,90,np.nan,88]
}

df=pd.DataFrame(data)
print("\nOriginal Data")
print(df)

# Drop rows with missing values
df_dropped=df.dropna()

print("\nAfter dropping missing values")
print(df_dropped)
