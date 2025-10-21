# train.py
import os
import argparse
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

def clean_text(text, stop_words):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

def main(args):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # input channel: training data path (SageMaker passes this)
    input_path = args.train_data
    df = pd.read_csv(input_path, usecols=['post', 'subreddit'])
    df = df.dropna()
    df = df.rename(columns={'post': 'text', 'subreddit': 'label'})
    df['text'] = df['text'].apply(lambda t: clean_text(t, stop_words))

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    model.fit(X, y)

    # Save both vectorizer and model together
    model_artifact = {
        'vectorizer': vectorizer,
        'model': model
    }

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model_artifact, os.path.join(args.model_dir, 'model.joblib'))
    print("Saved model to", os.path.join(args.model_dir, 'model.joblib'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='/opt/ml/input/data/train/mhd.csv')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--n-estimators', type=int, default=100)
    args = parser.parse_args()
    main(args)
