from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Sample training data
texts = ["I love this!", "This is amazing", "I hate it", "This is bad", "What a great experience", "Worst product ever"]
labels = [1, 1, 0, 0, 1, 0]  # 1 = positive, 0 = negative

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved.")
