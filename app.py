from flask import Flask, render_template, request, jsonify
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Load pre-trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['message']
    sentiment_score = sia.polarity_scores(user_input)['compound']
    vect_input = vectorizer.transform([user_input])
    prediction = model.predict(vect_input)[0]

    # Response logic
    if prediction == 1 and sentiment_score < -0.5:
        reply = ("It seems you're feeling low. Try journaling, deep breathing, or "
                 "talk to someone. Need help resources?")
    elif prediction == 1:
        reply = "You might be under stress. Would you like some relaxation tips?"
    else:
        reply = ("You're doing well! Stay positive, and let me know if you want "
                 "motivation or tips.")
    return jsonify({'reply': reply})

if __name__ == '__main__':
    # Use 0.0.0.0 to allow external connections if needed
    app.run(host='0.0.0.0', port=5000, debug=True)