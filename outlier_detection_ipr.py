import pandas as pd

data = {
    "salary": [20000, 25000, 30000, 35000, 100000]
}

df = pd.DataFrame(data)

Q1 = df["salary"].quantile(0.25)
Q3 = df["salary"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["salary"] < lower_bound) | (df["salary"] > upper_bound)]

print("Outliers:")
print(outliers)
