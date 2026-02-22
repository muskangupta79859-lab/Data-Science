# Feature Selection using SelectKBest

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# WITHOUT FEATURE SELECTION
# ------------------------------
model1 = LogisticRegression()
model1.fit(X_train_scaled, y_train)

pred1 = model1.predict(X_test_scaled)
acc1 = accuracy_score(y_test, pred1)

print("Accuracy without feature selection:", acc1)

# ------------------------------
# WITH FEATURE SELECTION
# ------------------------------
selector = SelectKBest(score_func=f_classif, k=2)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

model2 = LogisticRegression()
model2.fit(X_train_selected, y_train)

pred2 = model2.predict(X_test_selected)
acc2 = accuracy_score(y_test, pred2)

print("Accuracy with feature selection:", acc2)

print("Selected feature indices:", selector.get_support(indices=True))
