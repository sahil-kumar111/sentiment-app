from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os

app = Flask(__name__)

# Load the sentiment model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define labels
labels = ['Negative', 'Neutral', 'Positive']

# Function to analyze sentiment
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)[0]
    max_score, prediction = torch.max(probs, dim=0)
    return {
        "label": labels[prediction.item()],
        "score": round(max_score.item() * 100, 2)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    result = analyze_sentiment(review)
    sentiment = f"{result['label']} ({result['score']}%)"
    return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
