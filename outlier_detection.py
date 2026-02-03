import pandas as pd

# Sample Dataset
data = {
  "Age":[18,22,25,30,120,27,24,200],
  "Salary":[15000,20000,25000,30000,40000,28000,26000,1000000]
}
df = pd.DataFrame(data)

# IQR method for outlier detection
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers=df[
(df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
]

print("Outliers Detected:")
print(outliers)
