import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7],
    "Marks": [35, 40, 50, 60, 65, 70, 75]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied"]]
y = df["Marks"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# R2 Score
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.legend()
plt.show()
