import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Pipeline: scaling + model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=200))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "trained_model.pkl")
print("Model saved successfully!")

# Load model
loaded_model = joblib.load("trained_model.pkl")

# Predict new sample
sample = [X_test[0]]
prediction = loaded_model.predict(sample)

print("Prediction for sample:", prediction)
