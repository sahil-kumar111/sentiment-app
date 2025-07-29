from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run()
