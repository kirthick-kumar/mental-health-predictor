from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('mhd2.csv', usecols=['post', 'subreddit'])

# Drop rows where subreddit column equals the string 'subreddit' (likely header duplication)
df = df[df['subreddit'].str.lower() != 'subreddit']
df = df.dropna()

# Rename for consistency
df = df.rename(columns={'post': 'text', 'subreddit': 'label'})

# Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(clean_text)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_text = data.get("text", "")
    cleaned = clean_text(user_text)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)[0]
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
