from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Artificial imbalance create (demo purpose)
X = X[y != 2]
y = y[y != 2]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# BEFORE SMOTE
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
pred_before = model.predict(X_test)

print("Before SMOTE:\n", classification_report(y_test, pred_before))

# APPLY SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# AFTER SMOTE
model.fit(X_resampled, y_resampled)
pred_after = model.predict(X_test)

print("\nAfter SMOTE:\n", classification_report(y_test, pred_after))
