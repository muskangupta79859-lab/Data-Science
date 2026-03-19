import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load dataset
data = load_iris()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Convert to binary (for ROC)
y_bin = label_binarize(y, classes=[0,1,2])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict probabilities
y_score = model.predict_proba(X_test)

# ROC Curve for class 0
fpr, tpr, _ = roc_curve(y_test[:,0], y_score[0][:,0])
roc_auc = auc(fpr, tpr)

# Plot
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
