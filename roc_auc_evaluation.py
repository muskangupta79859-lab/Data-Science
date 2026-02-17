import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200))
])

# Train
model.fit(X_train, y_train)

# Probabilities for ROC-AUC (multi-class)
y_prob = model.predict_proba(X_test)
y_test_bin = label_binarize(y_test, classes=[0,1,2])

auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr")
print("ROC-AUC:", auc)

# Save model
joblib.dump(model, "roc_model.pkl")
print("Model saved")
