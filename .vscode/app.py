from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('HAPS_AI_Model.pkl')  

@app.route('/')
def home():
    return "💓 Heart Attack Prediction API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']  # [heart_rate, systolic, diastolic, spo2, ecg]
        if len(features) != 5:
            return jsonify({"error": "Expected 5 features"})

        prediction = model.predict([np.array(features)])
        return jsonify({
            "prediction": int(prediction[0]),
            "message": "1 = Heart attack risk, 0 = No risk"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
