from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Feature importance
importance = model.feature_importances_

# Table form me
result = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print(result)
