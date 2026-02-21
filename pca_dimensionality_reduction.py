# PCA Dimensionality Reduction + Model Comparison

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_iris()
X = data.data
y = data.target

# 2. Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# WITHOUT PCA
# -------------------------------
model_without_pca = LogisticRegression()
model_without_pca.fit(X_train_scaled, y_train)

pred_without = model_without_pca.predict(X_test_scaled)
acc_without = accuracy_score(y_test, pred_without)

print("Accuracy WITHOUT PCA:", acc_without)

# -------------------------------
# WITH PCA
# -------------------------------
pca = PCA(n_components=2)   # Reduce features to 2
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model_with_pca = LogisticRegression()
model_with_pca.fit(X_train_pca, y_train)

pred_with = model_with_pca.predict(X_test_pca)
acc_with = accuracy_score(y_test, pred_with)

print("Accuracy WITH PCA:", acc_with)

# -------------------------------
# Explained Variance
# -------------------------------
print("Explained variance ratio:", pca.explained_variance_ratio_)
