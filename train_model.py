from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample data (expand with PHQ-9 or GAD-7 dataset later)
texts = [
    "I feel hopeless", "I'm fine today", "I'm tired and sad", 
    "Everything is okay", "I can't sleep, feel stressed", 
    "I'm productive and happy"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = distress, 0 = normal

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = RandomForestClassifier()
model.fit(X, labels)

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')