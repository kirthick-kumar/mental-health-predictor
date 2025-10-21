# flask_app/app.py
from flask import Flask, request, jsonify
import joblib
import os

# RUN to make exe: chmod +x start.sh 
MODEL_PATH = os.environ.get('MODEL_PATH', '/opt/my-flask-app/model.joblib')

app = Flask(__name__)

# load model artifact
artifact = joblib.load(MODEL_PATH)
vectorizer = artifact['vectorizer']
model = artifact['model']

def clean_text(text):
    # replicate same cleaning as train.py (or import shared utils)
    import re, nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    user_text = data.get("text", "")
    cleaned = clean_text(user_text)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)[0]
    return jsonify({"prediction": str(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
