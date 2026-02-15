import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create pipeline
model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=200))
])

# Train model
model_pipeline.fit(X_train, y_train)

# Save model
joblib.dump(model_pipeline, "trained_model.pkl")
print("Model saved successfully")

# Load model
loaded_model = joblib.load("trained_model.pkl")
print("Model loaded successfully")

# Predict new data
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = loaded_model.predict(new_sample)

print("Prediction:", prediction)
