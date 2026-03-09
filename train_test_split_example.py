import pandas as pd
from sklearn.model_selection import train_test_split

data = {
    "hours_studied": [1,2,3,4,5,6,7,8],
    "sleep_hours": [8,7,6,6,5,5,4,4],
    "exam_score": [40,45,50,60,65,70,80,90]
}

df = pd.DataFrame(data)

X = df[["hours_studied", "sleep_hours"]]
y = df["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data")
print(X_train)

print("\nTesting Data")
print(X_test)
