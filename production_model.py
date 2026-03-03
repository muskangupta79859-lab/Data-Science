import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#  Load Dataset
data = load_iris()
X = data.data
y = data.target

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

# Train Model
pipeline.fit(X_train, y_train)

#  Evaluate
y_pred = pipeline.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

#  Save Model
joblib.dump(pipeline, "healthcare_model.pkl")
print("\nModel saved successfully!")

# Load Model
loaded_model = joblib.load("healthcare_model.pkl")

# Predict New Sample
sample = [X_test[0]]
prediction = loaded_model.predict(sample)
print("\nPrediction for sample:", prediction)
