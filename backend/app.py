from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from Baseline.build_model import train_model, get_unique_cyclist_ids, predict_cyclist_injury_probability, get_features_for_cyclist

app = Flask(__name__)
CORS(app)

model = train_model()
cyclist_ids = get_unique_cyclist_ids()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    cyclist_id = data['cyclist_id']
    probabilities = predict_cyclist_injury_probability(cyclist_id)
    features, dates = get_features_for_cyclist(cyclist_id)
    return jsonify({"probabilities": probabilities, "features": features, "dates": dates})

@app.route('/cyclists', methods=['GET'])
def get_cyclists():
    return jsonify(cyclist_ids)

if __name__ == '__main__':
    app.run(debug=True)
