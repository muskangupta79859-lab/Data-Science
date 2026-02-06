import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample dataset
data = {
    "Gender": ["Male", "Female", "Female", "Male"],
    "City": ["Delhi", "Mumbai", "Delhi", "Pune"]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)

#  Label Encoding 
label_encoder = LabelEncoder()
df["Gender_Label"] = label_encoder.fit_transform(df["Gender"])

print("\nAfter Label Encoding:")
print(df)

# One Hot Encoding 
df_onehot = pd.get_dummies(df, columns=["City"])

print("\nAfter One Hot Encoding:")
print(df_onehot)
