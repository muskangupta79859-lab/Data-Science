import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Sample Dataset
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Attendance": [60,65,70,75,80,85,90,95,88,92],
    "Pass": [0,0,0,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied", "Attendance"]]
y = df["Pass"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Hyperparameter Tuning
param_grid = {
    "knn__n_neighbors": [3,5,7]
}

grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)

# Best Model
best_model = grid.best_estimator_

# Prediction
y_pred = best_model.predict(X_test)

print("Best Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Cross Validation Score
cv_scores = cross_val_score(best_model, X, y, cv=5)
print("Cross Validation Score:", cv_scores.mean())
