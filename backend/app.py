from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

# Sample cyclist IDs
cyclist_ids = [1, 2, 3, 4, 5]

# Simulating the model with a dummy function
def get_injury_probabilities(cyclist_id):
    import numpy as np
    np.random.seed(cyclist_id)  # For reproducibility
    return [random.random() for _ in range(10)] # List of 30 probabilities

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    cyclist_id = data['cyclist_id']
    probabilities = get_injury_probabilities(cyclist_id)
    return jsonify(probabilities)

@app.route('/cyclists', methods=['GET'])
def get_cyclists():
    return jsonify(cyclist_ids)

if __name__ == '__main__':
    app.run(debug=True)
