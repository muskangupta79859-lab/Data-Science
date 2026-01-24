import pandas as pd
data={
  "Name":["Amit","Neha","Rahul",None],
  "Age":[20,21,None,22],
  "Marks":[85,None,78,90]
}
df=pd.DataFrame(data)

print("Original Data")
print(df)

print("\nMissing values count")
print(df.isnull().sum())

print("\nFilling missing values")
df["Age"].fillna(df["Age"].mean(),inplace=True)
df["Marks"].fillna(df["Marks"].median(),inplace=True)
df["Name"].fillna("Unknown",inplace=True)

print(df)
