from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/")
def home():
    return "Sentiment Model API Running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]
    
    text_vec = vectorizer.transform([data])
    prediction = model.predict(text_vec)[0]

    result = "Positive" if prediction == 1 else "Negative"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)
