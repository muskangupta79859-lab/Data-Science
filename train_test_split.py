import pandas as pd
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7],
    "Marks": [35, 40, 50, 60, 65, 70, 75]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied"]]   # Features
y = df["Marks"]             # Target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:")
print(X_train)

print("\nTesting Data:")
print(X_test)
