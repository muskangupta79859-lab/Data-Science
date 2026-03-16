import pandas as pd

data = {
    "city": ["Delhi", "Mumbai", "Chennai", "Delhi"]
}

df = pd.DataFrame(data)

encoded = pd.get_dummies(df["city"])

print("Original Data:")
print(df)

print("\nOne Hot Encoded Data:")
print(encoded)
