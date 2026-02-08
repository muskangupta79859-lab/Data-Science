import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7],
    "Marks": [35, 40, 50, 60, 65, 70, 75]
}

df = pd.DataFrame(data)

# Features & Target
X = df[["Hours_Studied"]]
y = df["Marks"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Predicted Marks:", y_pred)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
