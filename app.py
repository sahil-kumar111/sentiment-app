from flask import Flask, request, render_template
import joblib
    
app = Flask(__name__)
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['review']
    data = vectorizer.transform([text])
    prediction = model.predict(data)[0]
    label = "Positive 😊" if prediction == 1 else "Negative 😞"
    return render_template('index.html', prediction=label, review=text)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
