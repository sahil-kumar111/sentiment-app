from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    if review.strip() == "":
        sentiment = "Please enter a review."
        return render_template('index.html', sentiment=sentiment)

    transformed_review = vectorizer.transform([review])
    prediction = model.predict(transformed_review)[0]
    sentiment = 'Positive 😊' if prediction == 1 else 'Negative 😞'
    return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Use 10000 if no env port
    app.run(debug=False, host='0.0.0.0', port=port)
