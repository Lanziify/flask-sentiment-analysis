from flask import Blueprint, render_template, request, jsonify
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax

index_blueprint = Blueprint("index", __name__)


@index_blueprint.route("/")
def index():
    # Here, you can pass any data you want to the template
    index_data = {
        "title": "Welcome to Sentimental Comment Analyzer!",
        "message": "At Sentimental Comment Analyzer, we understand the power of words and the impact they can have on our lives. Our mission is to provide you with a tool that helps you understand the sentiment behind the comments you receive, whether they're on your social media posts, blog articles, or customer reviews.",
    }
    return render_template("index.html", data=index_data)


@index_blueprint.route("/analyze", methods=["POST"])
def sentiment():
    try:

        current_directory = os.path.dirname(__file__)

        tokenizer_path = os.path.join(current_directory, "..", "ml", "tokenizer")
        model_path = os.path.join(current_directory, "..", "ml", "model")

        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        model = BertForSequenceClassification.from_pretrained(model_path)

        encoded_input = tokenizer(
            request.form["comment"], padding=True, truncation=True, return_tensors="pt"
        )

        outputs = model(**encoded_input)
        probability_result = softmax(outputs[0][0].detach().numpy())
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        predicted_class = "Negative" if predicted_class == 0 else "Positive"

        probabilities = {
            "neg": round(probability_result[0], 6),
            "pos": round(probability_result[1], 6),
        }

        prediction_data = {
            "predictions": f'The comment was identified as {predicted_class}',
            "probabilities": probabilities,
        }

        return render_template("prediction.html", data=prediction_data), 200
    except Exception as e:
        print(e)
        return jsonify(msg="Error"), 500
