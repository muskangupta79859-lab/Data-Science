from flask import Flask, request, jsonify
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Demo: train + save model (agar file na ho)
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
joblib.dump(model, "model.pkl")

# Load model
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data]).tolist()
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
