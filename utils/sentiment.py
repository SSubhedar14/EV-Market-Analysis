# Utils.sentiment.py
# This module handles sentiment analysis using VADER and a pre-trained model.
import numpy as np
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer
from utils.preprocessing import preprocess_text, lemmatize_text

# Load VADER
sia = SentimentIntensityAnalyzer()

# Load model/vectorizer ONCE globally for efficiency
with open('models/lightgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def predict_sentiment_label(review_text):
    cleaned_text = preprocess_text(review_text)
    lemmatized_text = lemmatize_text(cleaned_text)
    tfidf_vector = vectorizer.transform([lemmatized_text]).toarray()
    vader_score = sia.polarity_scores(lemmatized_text)['compound']
    combined_vector = np.hstack((tfidf_vector, [[vader_score]]))
    _ = model.predict(combined_vector)[0]
    
    # Use VADER only for label (for clarity)
    if vader_score >= 0.05:
        return "Positive"
    elif vader_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def predict_numerical_score(review_text):
    cleaned_text = preprocess_text(review_text)
    lemmatized_text = lemmatize_text(cleaned_text)
    tfidf_vector = vectorizer.transform([lemmatized_text]).toarray()
    vader_score = sia.polarity_scores(lemmatized_text)['compound']
    combined_vector = np.hstack((tfidf_vector, [[vader_score]]))
    score = model.predict(combined_vector)[0]
    return int(np.clip(score, 1, 5))

def sentiment_breakdown(predictions):
    sentiment_count = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for p in predictions:
        if p in sentiment_count:
            sentiment_count[p] += 1
    total = sum(sentiment_count.values()) or 1
    return {k: round(v / total, 3) for k, v in sentiment_count.items()}

# ✅ Add this to allow external access from app.py
def load_model_and_vectorizer():
    return model, vectorizer
