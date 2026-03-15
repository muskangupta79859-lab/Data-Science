import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    "city": ["Delhi", "Mumbai", "Chennai", "Delhi", "Mumbai"]
}

df = pd.DataFrame(data)

encoder = LabelEncoder()

df["city_encoded"] = encoder.fit_transform(df["city"])

print(df)
