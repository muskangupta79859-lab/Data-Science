import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = load_iris()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")

# Load model
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Test prediction
prediction = loaded_model.predict(X[:5])
print("Sample Predictions:", prediction)
