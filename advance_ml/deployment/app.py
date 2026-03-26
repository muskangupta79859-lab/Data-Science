from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "ML Model API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']

    prediction = model.predict([features])

    return jsonify({
        'prediction': int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
