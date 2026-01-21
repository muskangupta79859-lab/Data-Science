# Basic Exploratory Data Analysis (EDA)

import pandas as pd
# Sample data
data={
  "Name":["A","B","C","D"],
  "Marks":[78,85,62,90],
  "Attendance":[85,90,70,95]
}
df=pd.DataFrame(data)

print("Dataset:")
print(df)

print("\nShape of dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nSummary statistics:")
print(df.describe())
