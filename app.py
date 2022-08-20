from nltk_utils import bag_of_words, tokenize
from flask import Flask, jsonify, request, make_response
import torch
import json
import random
from flask_cors import CORS, cross_origin
from model import NeuralNet

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('conversations.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "model_trained.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


@app.route('/', methods=['POST'])
@cross_origin()
def test_api_get():
    if request.method == 'POST':
        data = request.get_json()
        sentence = tokenize(data["text"])
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    result = {
                        "response": random.choice(intent['responses'])
                    }

                    return jsonify(result)

        result = {
            "response": "Sorry, I didn't quite catch that."
        }

        return jsonify(result)

app.run(debug=False)
