from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import random
# from Baseline.build_model import train_model, get_unique_cyclist_ids, predict_cyclist_injury_probability, get_features_for_cyclist

app = Flask(__name__)
CORS(app)

# model = train_model()
# cyclist_ids = get_unique_cyclist_ids()
# predict_cyclist_injury_proba = {}
# feature_for_cyclist = {}
# for cyclist_id in cyclist_ids:
#     predict_cyclist_injury_proba[cyclist_id] = predict_cyclist_injury_probability(cyclist_id)
#     feature_for_cyclist[cyclist_id] = get_features_for_cyclist(cyclist_id)





# load the data
with open('predict_cyclist_injury_proba.json') as f:
    predict_cyclist_injury_proba = json.load(f)

with open('feature_for_cyclist.json') as f:
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
