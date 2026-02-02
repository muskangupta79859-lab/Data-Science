import pandas as pd

data = {
    "Name": ["A", "B", "C", "D"],
    "Marks": [45, 78, 90, 60]
}

df = pd.DataFrame(data)

# Filtering
passed_students = df[df["Marks"] >= 60]
print("Passed Students:")
print(passed_students)

# Sorting
sorted_df = df.sort_values(by="Marks", ascending=False)
print("\nSorted Data:")
print(sorted_df)
