import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN Pipeline
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])

# Logistic Regression Pipeline
log_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression())
])

# Train
knn_pipeline.fit(X_train, y_train)
log_pipeline.fit(X_train, y_train)

# Predict
knn_pred = knn_pipeline.predict(X_test)
log_pred = log_pipeline.predict(X_test)

# Accuracy
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
