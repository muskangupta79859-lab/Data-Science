import pandas as pd

# Sample dataset
data = {
  "Marks":[45,50,55,60,65,70,200]
}
df = pd.DataFrame(data)

# Outlier Detection using IQR
Q1 = df["Marks"].quantile(0.25)
Q3 = df["Marks"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Replace Outliers with Median
median_value = df["Marks"].median()
median_value = df["Marks"].apply(lambda x: median_value if x < lower_bound or x > upper_bound else x)

print("Data after handling outliers:")
print(df)
