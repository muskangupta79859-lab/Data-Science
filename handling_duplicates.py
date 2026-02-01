import pandas as pd

data = {
    "Name": ["A", "B", "B", "C", "D", "D"],
    "Marks": [80, 75, 75, 90, 60, 60]
}

df = pd.DataFrame(data)

print("Original Data")
print(df)

print("\nDuplicate rows:")
print(df.duplicated())

df_cleaned = df.drop_duplicates()

print("\nAfter removing duplicates:")
print(df_cleaned)
