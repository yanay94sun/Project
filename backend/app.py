from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

# Define the base directory for file paths
base_dir = os.path.dirname(__file__)

# load the data
with open(os.path.join(base_dir, 'predict_cyclist_injury_proba.json')) as f:
    predict_cyclist_injury_proba = json.load(f)

with open(os.path.join(base_dir, 'feature_for_cyclist.json')) as f:
    feature_for_cyclist = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    cyclist_id = data['cyclist_id']
    probabilities, dict_x = predict_cyclist_injury_proba[str(cyclist_id)]
    features, dates = feature_for_cyclist[str(cyclist_id)]
    return jsonify({"probabilities": probabilities, "features": features, "dates": dates})

@app.route('/cyclists', methods=['GET'])
def get_cyclists():
    return jsonify(list(predict_cyclist_injury_proba.keys()))

if __name__ == '__main__':
    app.run(debug=True)
